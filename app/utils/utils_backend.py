import asyncio
from redis.asyncio import Redis
import pickle
from app.config import BACKEND_FASTAPI_LOG, dummy_model
from app.utils.utils_logging import initialize_logging, logger
from app.Ingestion_workflows.milvus_ingest import ingest2milvus
import os
import orjson
# logging config
initialize_logging(BACKEND_FASTAPI_LOG)

async def cleanup_expired_sessions(redis: Redis):
    """Clean up expired keys in redis to not overload.
    """
    while True:
        try:
            cursor = '0'
            logger.info("Initiating expired session cleanup in Redis")
            while cursor != 0:
                cursor, keys = await redis.scan(cursor, match="session:*", count=100)
                for key in keys:
                    ttl = await redis.ttl(key)
                    if ttl <= 0:
                        await redis.delete(key)
            
            # Wait for 3 hour before the next cleanup
            await asyncio.sleep(10800)
        except Exception as e:
            print(f"Error during session cleanup: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying if an error occurs

def deserialize(object):
    return pickle.loads(object)

def load_object(file):
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print("Error during unpickling object (Possibly unsupported):", e)

def session_id_get_question(chat_history, conv_id) -> tuple:
    """For a chat store of a conv_id, return a tuple of 1st question:convid to the frontend
    """
    conversation=chat_history.get("store", {})
    for chat_id, messages in conversation.items():
        user_blocks=messages[0].get("blocks", [])
        if not user_blocks:
            logger.warning("%s conv_id has an empty chat stored in db", conv_id)
            return ("ERROR, empty chat", conv_id)
        for user_text in user_blocks:
            if user_text.get("block_type") == "text":
                user_content = user_text.get("text", "")
    if ingest2milvus.get_token_len(dummy_model= dummy_model, text=user_content)>20: 
        try:
            user_content= (" ").join(user_content.split()[:9])+ "..."
        except Exception as e: 
            logger.warning("Error during getting previous question %s. Splitting by character instead %s", user_content, str(e))
            user_content=user_content + " "
            user_content= (" ").join(user_content.split()[:1])+ "..."
    return (user_content, conv_id)

def check_chat_history_db(db: str, user: str, user_conv_id: list, new_conv_id: str):
    """ Check chat history of user and return only those ids which are not empty chats (except for conv id assigned during current login)

    Args:
        db (str): path where chat stores are present
        user (str): current user
        user_conv_id (list): list of conv ids for the user which are not filtered
        new_conv_id (str): conv id assigned during current login

    Returns:
        final_conv_id_list: list of non empty conv ids
    """
    final_conv_id_list=[]
    for conv_id in user_conv_id:
        user_history= db.format(user=user, conv_id=conv_id)
        if os.path.exists(user_history):            
            with open(user_history, 'rb') as f:
                chat_history = orjson.loads(f.read())
            if chat_history["store"]:
                final_conv_id_list.append(session_id_get_question(chat_history=chat_history, conv_id=conv_id))
            elif conv_id==new_conv_id:
                final_conv_id_list.append(("New Chat", conv_id))
            elif not chat_history["store"]:
                logger.warning("%s is empty", user_history)
    return final_conv_id_list


def check_empty_chats(username: str, chat_store: str, id: str):
    """ Checks for empty chats or chat store not present for a user-session id pair

    Args:
        username (str): current user
        chat_store (str): chat store to check for empty chat
        id (str): current session id for which chat store is being checked

    Returns:
        bool: Returns True if the current session id is empty chat or does not have a chat store present
    """
    if os.path.exists(chat_store):  
        with open(chat_store, 'rb') as f:
            chat_history = orjson.loads(f.read())
        if not chat_history["store"]:
            os.remove(chat_store)
            logger.info("User %s had an empty session id: %s. Removing entry as logout initiated.", username, id)
            return True
    else:
        logger.info("session id: %s did not have a db entry. Removing it.", id)
        return True
    return False