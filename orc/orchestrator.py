import logging
import os
import sys
import openai
import openai.error
import time
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from azure.search.documents.models import QueryType
from datetime import datetime
from shared.util import get_chat_history_as_text, get_chat_history_as_messages, get_aoai_call_data, get_completion_text, prompt_tokens, format_answer

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)

# constants set from environment variables (external services credentials and configuration)

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT")
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE")

AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_GPT35TURBO_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT35TURBO_DEPLOYMENT") or "chat"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"

AZURE_DB_KEY = os.environ.get("AZURE_DB_KEY")
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER")

# prompt files
QUERY_STRING_PROMPT_FILE = f"orc/prompts/query_string.txt" 
QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.txt"
ANSWER_ENRICHMENT_PROMPT_FILE = f"orc/prompts/answer_enrichment.txt"

# predefined answers
THROTTLING_ANSWER = "Lo siento, nuestros servidores están demasiado ocupados, por favor inténtelo de nuevo en 10 segundos."
ERROR_ANSWER = "Lo siento, tuvimos un problema con la solicitud."

# misc
NUM_SOURCES_FROM_SEARCH = 3
ANSWER_MAX_TOKENS = 1024
CHATGPT_TIKTOKEN_ENCODER = "cl100k_base"
QUERY_LANGUAGE = "es-ES"
BLOCKED_SOURCES_FILE = f"orc/misc/blocked_sources.txt"
ANSWER_FORMAT = "html" # html, markdown, none


def run(conversation_id, ask):

    # 1) Create and configure service clients (cosmos, search and openai)

    # db
    db_client = CosmosClient(AZURE_DB_URI, credential=AZURE_DB_KEY, consistency_level='Session')
    # search
    search_creds = AzureKeyCredential(AZURE_SEARCH_KEY)
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=search_creds)
    # openai
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
    openai.api_version = "2023-03-15-preview" 
    openai.api_key = AZURE_OPENAI_KEY

    # 2) Get conversation stored in CosmosDB

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    # state mgmt
    previous_state = "none"
    current_state = "none"

    # get conversation
    db = db_client.create_database_if_not_exists(id=AZURE_DB_ID)
    container = db.create_container_if_not_exists(id=AZURE_DB_CONTAINER, partition_key=PartitionKey(path='/id', kind='Hash'))
    try:
        conversation = container.read_item(item=conversation_id, partition_key=conversation_id)
        previous_state = conversation.get('state')
    except Exception as e:
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} customer sent an inexistent conversation_id, create new conversation_id")        
        conversation = container.create_item(body={"id": conversation_id, "state": previous_state})
    logging.info(f"[orchestrator] {conversation_id} previous state: {previous_state}")

    # history
    history = conversation.get('history', [])
    history.append({"role": "user", "content": ask})
    # conversation_data
    conversation_data = conversation.get('conversation_data', 
                                    {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': [], 'aoai_calls': []})
    # transaction_data
    transactions = conversation.get('transactions', [])

    # 3) Define state/intent (money_transfer,question_answering,...) based on conversation and last statement from the user 

    # TODO: when working with transactional scenarios implement a triage function to define state based on conversation and last statement from the user
    current_state = "question_answering" # let's stick to question answering for now
    conversation['state'] = current_state
    conversation = container.replace_item(item=conversation, body=conversation)

    # 4) Use conversation functions based on state

    # Initialize iteration variables
    answer = "none"
    sources = "none"
    search_query = "none"
    transaction_data_json = {}

    if current_state == "question_answering":
    # 4.1) Question answering

        # generating an optimized keyword search query based on the chat history and the last question
                
        start_time = time.time()

        # qna prompts
        query_string_prompt = open(QUERY_STRING_PROMPT_FILE, "r").read()

        seach_results = ""

        prompt = query_string_prompt.format(chat_history=get_chat_history_as_text(history, include_last_turn=False), question=ask)
        completion_enriched = openai.Completion.create(
            engine=AZURE_OPENAI_GPT_DEPLOYMENT, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=100, 
            n=1, 
            stop=["\n"])
        search_query = completion_enriched.choices[0].text           
        conversation_data['aoai_calls'].append(get_aoai_call_data(prompt, completion_enriched))
        
        response_time = time.time() - start_time
        logging.info(f"[orchestrator] generated query string with gpt. {response_time} seconds")


        # searching documents

        start_time = time.time()

        search_result = search_client.search(search_query, 
                                top=50,
                                query_type=QueryType.SEMANTIC, 
                                query_language=QUERY_LANGUAGE,   
                                query_speller="lexicon", 
                                semantic_configuration_name="default")
        
        # optional: get blocked sources from config file
        blocked_sources = []
        if os.path.exists(BLOCKED_SOURCES_FILE):
            with open(BLOCKED_SOURCES_FILE, 'r') as file:
                blocked_sources = [file.strip() for file in file.readlines()]

        # build sources list
        documents = []
        for doc in search_result:
            blocked = False
            for blocked_word in blocked_sources:
                if blocked_word in doc[KB_FIELDS_SOURCEPAGE]: blocked = True
            if not blocked:
                documents.append(doc[KB_FIELDS_SOURCEPAGE] + ": "+ doc[KB_FIELDS_CONTENT].strip() + "\n")
            if len(documents) >= NUM_SOURCES_FROM_SEARCH: break
        seach_results = documents

        response_time = time.time() - start_time
        logging.info(f"[orchestrator] searched for documents. {response_time} seconds")


        # create question answering prompt

        prompt = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read() 

        history_messages=get_chat_history_as_messages(history, include_last_turn=True)
        # calibrates the number of sources based on model's token limit
        if len(seach_results) > 0:
            # question answering prompt with search_results
            token_limit = int(os.environ.get("AZURE_OPENAI_CHATGPT_LIMIT")) - ANSWER_MAX_TOKENS
            num_tokens = sys.maxsize
            while num_tokens > token_limit:
                sources = "\n".join(seach_results)
                prompt = prompt.format(sources=sources)
                messages = [
                    {"role": "system", "content": prompt}   
                ]
                messages = messages + history_messages
                num_tokens = prompt_tokens(messages, CHATGPT_TIKTOKEN_ENCODER)

        else:
            sources = "\n There are no sources for this question, say you don't have the answer"
            prompt = prompt.format(sources=sources)
            messages = [
                {"role": "system", "content": prompt}   
            ]
            messages = messages + history_messages


        # calling gpt model to get the answer

        start_time = time.time()

        error = False        
        try:
            completion = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                messages=messages,
                temperature=0.0,
                max_tokens=ANSWER_MAX_TOKENS
            )
            answer = completion['choices'][0]['message']['content']
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))
        except openai.error.RateLimitError as e:
            answer = THROTTLING_ANSWER
            error = True
        except openai.error.InvalidRequestError as e:
            error_message = str(e)
            answer = f'{ERROR_ANSWER}. {error_message}'       
            error = True

        response_time = time.time() - start_time
        logging.info(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")
        if not error:
            logging.info(f"[orchestrator] called gpt model to get the answer. {completion.usage.prompt_tokens} prompt tokens. {completion.usage.completion_tokens} completion tokens.")

        # optional: answer enrichment

        if not error and os.path.isfile(ANSWER_ENRICHMENT_PROMPT_FILE):
            start_time = time.time()
        
            enrichment_prompt = open(ANSWER_ENRICHMENT_PROMPT_FILE, "r").read()
            messages_enriched = [{"role": "system", "content": enrichment_prompt},
                                 {"role": "user", "content": answer}]
            completion_enriched = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_GPT35TURBO_DEPLOYMENT,
                messages=messages_enriched,
                temperature=0.8,
                max_tokens=ANSWER_MAX_TOKENS
            )
            answer =  get_completion_text(completion_enriched)
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages_enriched, completion_enriched))

            response_time = time.time() - start_time
            logging.info(f"[orchestrator] called gpt model to enrich the answer. {response_time} seconds")

    # 6. update and save conversation (containing history and conversation data)

    history.append({"role": "assistant", "content": answer})
    conversation['history'] = history
   
    conversation_data['interactions'].append({'user_message': ask, 'previous_state': previous_state, 'current_state': current_state})
    conversation['conversation_data'] = conversation_data

    conversation['transactions'] = transactions

    conversation = container.replace_item(item=conversation, body=conversation)

    # 7. return answer

    result = {"conversation_id": conversation_id, 
              "answer": format_answer(answer, ANSWER_FORMAT), 
              "current_state": current_state, 
              "data_points": sources, 
              "thoughts": f"Searched for:\n{search_query}\n\nPrompt:\n{prompt}",
              "transaction_data": transaction_data_json}
    return result