from config_URL import MILVUS_URI, API_GET_EXISTING_CONV, API_GET_AVAILABLE_DOC_NAMES, API_LOG_FEEDBACK
from pymilvus import MilvusClient
from utils.utils_logging import logger, initialize_logging, FRONTEND_LOG # noqa: E402
import json
import pandas as pd
import requests

# logging config
initialize_logging(FRONTEND_LOG)

# Helper functions
def show_client(username, password):
    """ Return client for user login
    """
    return MilvusClient(uri=MILVUS_URI, token=f"{username}:{password}")

def feedback_logger(**kwargs):
    """ Send API call to log feedback for a session and LLM response in DB

    Returns:
        True if feedback was logged successfully, else false
    """
    data= kwargs
    logger.info(data)
    response_feedback = requests.post(
            API_LOG_FEEDBACK,
            json=data
        )
    response_feedback.raise_for_status()
    return response_feedback.json()

def get_all_sessions(username: str, new_conv_id: str) -> list:
    """ Get all sessions from redis conversation tracker followed by filtering on which ones are actually present in chat history db

    Args:
        username (str): user for whom conversation ids need be retrieved

    Returns:
        list: list of existing conversation ids
    """
    session_id_response = requests.get(API_GET_EXISTING_CONV.format(session_id_user=new_conv_id + "$" +username))
    session_id_response.raise_for_status()
    existing_sessions = session_id_response.json()
    return existing_sessions

def convert_chat(orig_chat: str):
    """ Convert the llamaindex chat history to gradio chatbot compatible chat format

    Args:
        orig_chat (str): chat history retrieved

    Returns:
        gradio_messages: gradio chat history for selected conv id
    """
    gradio_messages = []
    orig_chat= json.loads(orig_chat) # convert str from api to dict 
    conversation = orig_chat.get("store", {})
    # Iterate through the conversation
    for chat_id, messages in conversation.items():
        for message in messages:
            role = message.get("role")
            
            # Extract text from blocks list
            blocks = message.get("blocks", [])
            content = ""
            for block in blocks:
                if block.get("block_type") == "text":
                    content = block.get("text", "")
            # Add to gradio messages format
            if content:
                gradio_messages.append({"role": role, "content": content})
    return gradio_messages

def make_doc_markdown(available_doc_list):
    """ Markdown creation fucntion

    Args:
        available_doc_list (list): list of dict with keys as documents available in DB for the user

    Returns:
        doc_markdown: str
    """
    doc_list=[]
    if not available_doc_list:
        return pd.DataFrame.from_dict({"Document": ["Documents not found"], "Date of Creation": [""]})
    # doc_markdown= "## The available documents you can chat with are: <br>"
    for index, doc in enumerate(available_doc_list, start=1):
        doc_list.append({"Document": list(doc.keys())[0], "Date of Creation":doc[list(doc.keys())[0]]})
    df= pd.DataFrame(doc_list)
    
    return df

def get_doc_names_frontend(conv_id: str, username: str):
    """ Gets document names available for the user and creates a markdown to display them in frontend. 
    Also for the user, get status of admin access privileges.

    Args:
        conv_id (str): conv id of user
        username (str): username

    Returns:
        available_doc_markdown(str): Markdown of available document info to show in the frontend
    """
    available_docs_response= requests.get(API_GET_AVAILABLE_DOC_NAMES.format(session_id_user=conv_id + "$" + username))
    available_docs_response.raise_for_status()
    available_docs= available_docs_response.json()[0]
    admin_access_user= available_docs_response.json()[1]
    collection_type = available_docs_response.json()[2]
    try:
        available_doc_markdown= make_doc_markdown(available_docs)
    except Exception as e: 
        logger.error("Error while retrieving doc names for user %s: %s", username, str(e))
    return available_doc_markdown, admin_access_user, collection_type