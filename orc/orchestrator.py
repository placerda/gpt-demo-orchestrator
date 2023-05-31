import json
import logging
import os
import openai
import openai.error
import re
import tiktoken
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey
from azure.search.documents.models import QueryType
from datetime import datetime

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)

# constants

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT")
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE")

AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")

AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_GPT35TURBO_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT35TURBO_DEPLOYMENT") or "chat"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"
OAI_CHATGPT_ENCODER = "cl100k_base"

AZURE_DB_URI = os.environ.get("AZURE_DB_URI")
AZURE_DB_KEY = os.environ.get("AZURE_DB_KEY")
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER")

PROMPT_FOLDER = "orc/prompts"
NUM_SOURCES = 15

## utility functions

def prompt_tokens(prompt, encoding_name: str) -> int:
    num_tokens = 0
    # convert prompt to string when it is a list
    if isinstance(prompt, list):
        messages = prompt
        prompt = ""
        for m in messages:
            prompt += m['role']
            prompt += m['content']
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

def get_chat_history_as_text(history, include_last_turn=True, approx_max_tokens=1000):
    history_text = ""
    if len(history) == 0:
        return history_text
    for h in reversed(history if include_last_turn else history[:-1]):
        # history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
        history_text = f"""<|im_start|>{h["role"]}""" +"\n" + h["content"] + "\n" + """<|im_end|>""" + "\n" + history_text
        if len(history_text) > approx_max_tokens*4:
            break    
    return history_text

def get_completion_text(completion):
    if 'text' in completion['choices'][0]:
        return completion['choices'][0]['text'].strip()
    else:
        return completion['choices'][0]['message']['content'].strip()

def get_aoai_call_data(prompt, completion):
    prompt_words = 0
    if isinstance(prompt, list):
        messages = prompt
        prompt = ""
        for m in messages:
            prompt += m['role'].replace('\n',' ')
            prompt += m['content'].replace('\n',' ')
        prompt_words = len(prompt.split())
    else:
        prompt = prompt.replace('\n',' ')
        prompt_words = len(prompt.split())

    return {"model": completion.model, "prompt_tokens": completion.usage.prompt_tokens, "prompt_words": prompt_words,
            "completion_words": len(get_completion_text(completion).split()), "completion_tokens": completion.usage.completion_tokens}

def extract_sentiment(history):
    messages = [{"role": "system", "content": "You are an AI assistant that helps define the sentiment of a conversation. You are not allowed to ask questions or make statements. You can only respond with a sentiment"},]
    if len(messages) > 1: messages.append(history[-2])
    if len(messages) > 0: messages.append(history[-1])
    messages.append({"role": "user", "content": "say in one word in small letters what is the sentiment of this conversation [positive,neutral,negative] dont use punctuation"})
    completion = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_GPT35TURBO_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=50
    )
    sentiment = re.sub(r'\W+', '', get_completion_text(completion)).lower()
    aoai_call_record = get_aoai_call_data(messages, completion)
    return sentiment, aoai_call_record

# main function

def run(conversation_id, question):

    # 1) Create service clients (cosmos, search and openai)

    db_client = CosmosClient(AZURE_DB_URI, credential=AZURE_DB_KEY, consistency_level='Session')
    search_creds = AzureKeyCredential(AZURE_SEARCH_KEY)
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=search_creds)
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
    openai.api_version = "2023-03-15-preview" 
    openai.api_key = AZURE_OPENAI_KEY

    # 2) Load prompts from files

    # general prompts 
    main_chat_prompt = open(f"{PROMPT_FOLDER}/general_main_prompt.txt", "r").read()

    # qna prompts
    query_string_prompt = open(f"{PROMPT_FOLDER}/qna_query_string.txt", "r").read()
    answer_prompt = open(f"{PROMPT_FOLDER}/qna_answer.txt", "r").read()

    # transactional prompts
    state_extraction_prompt = open(f"{PROMPT_FOLDER}/general_state_extraction.txt", "r").read()
    money_transfer_prompt = open(f"{PROMPT_FOLDER}/trans_money_transfer.txt", "r").read()
    extract_money_transfer_data_prompt = open(f"{PROMPT_FOLDER}/trans_extract_money_transfer_data.txt", "r").read()
    check_if_finished_prompt = open(f"{PROMPT_FOLDER}/trans_check_if_finished.txt", "r").read()

    # 3) Get conversation context and its content

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    # state mgmt
    previous_state = "none"
    current_state = "none"

    # get conversation context
    db = db_client.create_database_if_not_exists(id=AZURE_DB_ID)
    container = db.create_container_if_not_exists(id=AZURE_DB_CONTAINER, partition_key=PartitionKey(path='/id', kind='Hash'))
    try:
        context = container.read_item(item=conversation_id, partition_key=conversation_id)
        previous_state = context.get('state')
    except Exception as e:
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} customer sent an inexistent conversation_id, create new conversation_id")        
        context = container.create_item(body={"id": conversation_id, "state": previous_state})
    logging.info(f"[orchestrator] {conversation_id} previous state: {previous_state}")

    # history
    history = context.get('history', [])
    history.append({"role": "user", "content": question})
    # conversation_data
    conversation_data = context.get('conversation_data', 
                                    {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': [], 'aoai_calls': []})
    # transaction_data
    transactions = context.get('transactions', [])

    # 4) Define state (money_transfer,question_answering,...) based on conversation and last statement from the user 
    current_state = "question_answering"
    
    # prompt = state_extraction_prompt.format(chat_history=get_chat_history_as_text(history, include_last_turn=False), question=question)
    # try:
    #     completion = openai.Completion.create(
    #         engine=AZURE_OPENAI_GPT_DEPLOYMENT, 
    #         prompt=prompt, 
    #         temperature=0.0, 
    #         max_tokens=100, 
    #         n=1, 
    #         stop=["\n"])
    #     current_state = get_completion_text(completion).lower()
    #     conversation_data['aoai_calls'].append(get_aoai_call_data(prompt, completion))
    # except Exception as e:
    #     current_state = "could_not_define"
    # current_state = "question_answering" if current_state == "could_not_define" else current_state
    # logging.info(f"[orchestrator] {conversation_id} current state: {current_state}")

    context['state'] = current_state
    context = container.replace_item(item=context, body=context)

    # 5) Use conversation prompts based on state

    # Initialize iteration variables
    answer = "none"
    sources = "none"
    search_query = "none"
    transaction_data_json = {}

    # 5.1) Money Transfer
    if current_state == "money_transfer":
        
        prompt = main_chat_prompt.format(sub_prompt=money_transfer_prompt, sources="")

        messages = [
            {"role": "system", "content": prompt}   
        ]
        messages = messages + history
        try:
            completion = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                messages=messages,
                temperature=0.0,
                max_tokens=1024
            )
            answer = completion['choices'][0]['message']['content']
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))
        except openai.error.RateLimitError as e:
            answer = 'Lo siento, nuestros servidores están demasiado ocupados, por favor inténtelo de nuevo en 10 segundos.'
    
        # Transaction data
        transaction_data_json = {}
        extract_money_transfer_data_prompt = extract_money_transfer_data_prompt.format(messages=get_chat_history_as_text(history))
        completion = openai.Completion.create(
            engine=AZURE_OPENAI_GPT_DEPLOYMENT, 
            prompt=extract_money_transfer_data_prompt, 
            temperature=0.3,
            max_tokens=350,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=None)
        transaction_data = completion.choices[0].text.strip().lower()
        conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))
        try:
            transaction_data_json = json.loads(transaction_data)
        except Exception as e:
            pass

        # Check if transaction is finished
        prompt_is_finished = check_if_finished_prompt.format(transaction=current_state, chat_history=get_chat_history_as_text(history), answer=answer)
        completion = openai.Completion.create(
            engine=AZURE_OPENAI_GPT_DEPLOYMENT, 
            prompt=prompt_is_finished, 
            temperature=0.0, 
            max_tokens=100, 
            n=1, 
            stop=["\n"])
        is_finished = True if completion.choices[0].text.strip().lower() == "transaction_is_finished" else False
        conversation_data['aoai_calls'].append(get_aoai_call_data(prompt_is_finished, completion))
        if is_finished:
            transaction_data_json['is_finished'] = True
            transactions.append(transaction_data_json)
            # current_state = "question_answering"
            logging.info(f"[orchestrator] {conversation_id} transaction finished")
        else:
            transaction_data_json['is_finished'] = False            
            logging.info(f"[orchestrator] {conversation_id} transaction not finished")


    else:
    # 5.2) Question answering

        seach_results = ""
        # Create an optimized keyword search query based on the chat history and the last question
        prompt = query_string_prompt.format(chat_history=get_chat_history_as_text(history, include_last_turn=False), question=question)
        completion = openai.Completion.create(
            engine=AZURE_OPENAI_GPT_DEPLOYMENT, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=100, 
            n=1, 
            stop=["\n"])
        search_query = completion.choices[0].text           
        conversation_data['aoai_calls'].append(get_aoai_call_data(prompt, completion))
        # Retrieve relevant documents from the search index with the GPT optimized query 
        NUM_ANSWERS = 1

        r = search_client.search(search_query, 
                                top=NUM_SOURCES,
                                query_type=QueryType.SEMANTIC, 
                                query_language="es-ES",   
                                query_speller="lexicon", 
                                semantic_configuration_name="default",
                                query_answer="extractive",
                                query_answer_count=NUM_ANSWERS)
        
        # build sources list
        top_document_page = ""
        documents = []
        for doc in r:
            if top_document_page=="": top_document_page = doc[KB_FIELDS_SOURCEPAGE]
            documents.append(doc[KB_FIELDS_SOURCEPAGE] + ": "+ doc[KB_FIELDS_CONTENT].strip() + "\n")
        search_answers = []

        # stopped using search answers because it may not have the proper context to answer the question
        # try:
        #     for answer in r.get_answers():
        #         if answer.text != "":
        #             search_answers.append(top_document_page + ": " + answer.text.strip() + "\n")
        # except TypeError:
        #     pass

        seach_results = search_answers + documents

        # Generate a contextual and content specific answer using the search results and chat history

        answer_max_tokens = 1024

        # prompt generation

        if len(seach_results) > 0:
            # question answering prompt with search_results (calibrates the number of sources based on token limit)
            token_limit = int(os.environ.get("AZURE_OPENAI_CHATGPT_LIMIT")) - answer_max_tokens
            num_tokens = 1000000
            while num_tokens > token_limit:
                seach_results = seach_results[:-1]
                sources = "".join(seach_results)
                prompt = main_chat_prompt.format(sub_prompt=answer_prompt, sources=sources)
                messages = [
                    {"role": "system", "content": prompt}   
                ]
                messages = messages + history 
                num_tokens = prompt_tokens(messages, OAI_CHATGPT_ENCODER)

        else:
            sources = "\n There are no sources for this question, say you don't have the answer"
            prompt = main_chat_prompt.format(sub_prompt=answer_prompt, sources=sources)
            messages = [
                {"role": "system", "content": prompt}   
            ]
            messages = messages + history

        # answer generation
        
        try:
            completion = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                messages=messages,
                temperature=0.0,
                max_tokens=answer_max_tokens
            )
            answer = completion['choices'][0]['message']['content']
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))
        except openai.error.RateLimitError as e:
            answer = 'Lo siento, nuestros servidores están demasiado ocupados, por favor inténtelo de nuevo en 10 segundos.'

    # 6. enrich conversation history

    # sentiment, aoai_call_record = extract_sentiment(history)
    # conversation_data['aoai_calls'].append(aoai_call_record)
    sentiment = "none"

    # 7. update and save conversation context (containing history and conversation data)

    history.append({"role": "assistant", "content": answer})
    context['history'] = history
   
    conversation_data['interactions'].append({'user_message': question, 'previous_state': previous_state, 'current_state': current_state, 'sentiment': sentiment})
    context['conversation_data'] = conversation_data

    context['transactions'] = transactions

    context = container.replace_item(item=context, body=context)

    # 8. return answer

    result = {"conversation_id": conversation_id, 
              "answer": answer, 
              "current_state": current_state, 
              "data_points": sources, 
              "thoughts": f"Searched for:\n{search_query}\n\nPrompt:\n{prompt}",
              "transaction_data": transaction_data_json}
    return result