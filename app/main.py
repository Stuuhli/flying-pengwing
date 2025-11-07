from app.Ingestion_workflows.docling_parse_process import Docling_parser
from app.Ingestion_workflows.milvus_ingest import ingest2milvus
from app.RAG_workflows.citation_engine import CitationQueryEngineWorkflow
from app.auth import password_verify
from app.config import (USER_DB_PATH, MILVUS_URI, TOKEN,BACKEND_FASTAPI_LOG, RETRIEVAL_LOG_PATH, USER_HISTORY, USER_COLLECTION_MAPPING, CHAT_STORE_PATH,
                         MILVUS_ROOT_ROLE, BACKEND, VLLM_GEN_URL, GEN_CONTEXT_WINDOW, FILES_DB, FASTAPI_URL, col_mod, topk_mod, dim_mod, collection_type,
                           systemprompt, citation_header)
from app.utils.utils_LLM import milvus_hybrid_retrieve, cite, log_retrievals 
from app.utils.utils_auth import user_auth_format, write_json, load_json, user_auth_validate
from app.utils.utils_backend import deserialize, cleanup_expired_sessions, check_chat_history_db, check_empty_chats
from app.utils.utils_ingestion import FileUploadValidator, milvus_db_as_excel, ingest, get_doc_in_collection, check_admin
from app.utils.utils_logging import initialize_logging, logger
from app.utils.utils_req_templates import session_start_req, Message_request, Ingest_req, change_col, Logout_req, change_session, feedback_model
from contextlib import asynccontextmanager
from fastapi import Depends
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_utils.timing import add_timing_middleware
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from pymilvus import MilvusClient
from redis.asyncio import Redis
from typing import Dict
import asyncio
import copy
import httpx
import json
import os
import pickle
import shutil
import pandas as pd

# logging config
initialize_logging(BACKEND_FASTAPI_LOG)

#global variables
MODEL = os.getenv("MODEL", "llama3.2")
validator = FileUploadValidator(max_size_mb=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ On starting up of the FastAPI server, this function initializes the ollama model and the user history tracker variable and file
    """
    logger.info(f" Initializing {BACKEND} and Redis")
    # decode responses is False as manual decoding is necessary for some of the redis "get" commands
    app.state.redis = Redis(host='localhost', port=6379, db=0, decode_responses=False)
    if BACKEND=="ollama":
        Settings.llm = Ollama(model=MODEL, request_timeout=150.0, temperature=0)
    elif BACKEND=="vllm":
        Settings.llm = OpenAILike(model=MODEL, api_base=VLLM_GEN_URL, api_key="random", temperature=0.1, timeout=180.0, context_window=GEN_CONTEXT_WINDOW, is_chat_model=True)
    user_history_db= Docling_parser.get_store(path=USER_HISTORY)
    # set chat store path and user-session tracker
    await app.state.redis.set("chat_store_path", CHAT_STORE_PATH)
    await app.state.redis.set("conversations", json.dumps(user_history_db))
    asyncio.create_task(cleanup_expired_sessions(app.state.redis))
    yield
    await app.state.redis.aclose()

app = FastAPI(
    lifespan = lifespan
)

# Change this
""" It allows cross origin resource sharing between backend and frontend """
origins= [
    FASTAPI_URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_timing_middleware(app, record=logger.info, prefix="timing")
app.mount("/doc_store", StaticFiles(directory="/doc_store"), name="doc_store")

@app.get("/")
def welcome() -> Dict[str, str]:
    """ On startup of the server, we get this
    """
    return {"hello": "there"}

@app.get("/health")
def health() -> Dict[str, str]:
    """ We can check if server is working if this function works
    """
    return {"health": "ok"}


@app.post("/create_user")
async def create_user(request : user_auth_format):
    """ Create new user account by putting the data sent to API into an existing json using write_json function
    """
    logger.info(" Creating user: %s", request.username)
    user_details= {"username": request.username,
        "full_name": request.fullname,
        "hashed_password": request.password,
        "disabled": request.disabled,
        "admin": request.admin}
    write_json(username=request.username, new_data= user_details, filename=USER_DB_PATH)
    logger.info(" User successfully created in DB: %s", request.username)


@app.post("/validate_user/{user}")
async def validate_user(request: user_auth_validate):
    # can also check if user exists in milvus, if not, create role if exist in user db
    """ Validate username and entered password with the exisiting db. 
    """
    logger.info(" Validating: %s", request.username)
    user_db= load_json(filename=request.USER_DB_PATH)
    if request.username not in user_db:
            logger.info("Invalid username or password ")
            return "Invalid username or password", False
    if not password_verify(password=request.password, hashed= user_db[request.username]["hashed_password"]):
            logger.info("Invalid username or password ")
            return "Invalid username or password", False
    logger.info(" Validation successful for %s", request.username)
    return "Successfully Authenticated.", True

@app.post("/conversation/start")
async def start_conversation(request: session_start_req, redis: Redis = Depends(lambda: app.state.redis)):
    """ This function does initializations at the start of conversation. 
        The state variables that are stored by redis include:
        1. User history path and chat store paths
        2. conversations tracker which is updated at the start and updates the db stored in user history path. It keeps track of which user-conversation id is the current api call for
        3. session data: for a particular session id, store as a hash the user, passwrd, chat store object and citation engine object for later use
    """
    logger.info(" Initializing session: %s for user %s", request.conv_id, request.username)
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)
    user_collection_db= Docling_parser.get_store(path=USER_COLLECTION_MAPPING)
    list_of_conversations=[conv_id for conversations in conversations_dict.values() for conv_id in conversations]
    if request.conv_id in list_of_conversations:
        logger.error(f"Invalid session: {request.conv_id}.  Already present in DB")
        raise HTTPException(status_code=401, detail="Invalid conv id. Already present in DB.")
    # Initialize the chat store 
    chat_store = SimpleChatStore()
    # initialise memory buffer and serialise it
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=14000,
        chat_store=chat_store,
        chat_store_key=request.conv_id,
    )
    serialized_memory = pickle.dumps(memory)
    # store for each session, the username, password, chatstore and memory in redis
    # TODO password encryption to make it secure
    session_data= {
        "user_id": request.username,
        "password": request.password,
        "memory": serialized_memory,
        "read_collection": user_collection_db[request.username],
        "ingest_collection": ""
    }
    await redis.hset(f"session:{request.conv_id}",mapping= session_data)
    await redis.expire(f"session:{request.conv_id}", 10800)
    logger.info("Successfully set up session variables in Redis for session: %s ", request.conv_id)
    # persist the new chat store for the user in the appropriate directory with name as conv id
    deserialized_chat_store= deserialize(serialized_memory)
    deserialized_chat_store= deserialized_chat_store.chat_store
    chat_store_path= await redis.get("chat_store_path")
    chat_store_path= chat_store_path.decode('utf-8')
    deserialized_chat_store.persist(persist_path=chat_store_path.format(user= request.username, conv_id= request.conv_id))
    # check if new user, then create new entry in conversations dict and finally append the new session id
    if request.username not in conversations_dict:
        conversations_dict[request.username]=[]
    conversations_dict[request.username].append(request.conv_id)
    await redis.set("conversations", json.dumps(conversations_dict))
    # persist the conversation tracker
    with open(USER_HISTORY, 'w') as fp:
        json.dump(conversations_dict, fp)
    logger.info("Started session: %s", request.conv_id)
    return {"message": f"Conversation {request.conv_id} started", "user_collection": user_collection_db[request.username]}

@app.post("/conversation/{session_id_user}/message")
async def add_message(session_id_user:str, request: Message_request, redis: Redis = Depends(lambda: app.state.redis)) -> str:
    """ This function for a particular conversation id, sends the message to the "Chat engine" LLM object which generates the response and updates the chat memory.
        Then it returns the response back to API and saves the updated chat store locally. 
    """
    logger.info(" User message to bot for session: %s, message:%s", request.conv_id, request.message)
    user_name_current= session_id_user.split("$")[1]
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)
    if request.conv_id not in conversations_dict[user_name_current]:
        logger.error(f"Conversation not found for session: {request.conv_id}", exc_info=True)
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        # note: change TOKEN to password from user to login into milvus client
        collection= await redis.hget(f"session:{request.conv_id}", "read_collection")
        collection= collection.decode('utf-8')
        top_k= topk_mod[col_mod[collection]]
        retrievals, results = await milvus_hybrid_retrieve(uri=MILVUS_URI,token=TOKEN, question=request.message, collection_name=collection, model= col_mod[collection], k=top_k)
        logger.info("For session: %s and collection: %s, %s relevant sources found", request.conv_id, collection, len(results))
        # get responses after initialising engine with latest memory for the session
        memory= await redis.hget(f"session:{request.conv_id}", "memory")
        if not memory:
            logger.error(f"Invalid or expired session: {request.conv_id}")
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        memory= deserialize(memory)
        model = CitationQueryEngineWorkflow(LLM=Settings.llm, memory=memory, system_prompt=systemprompt[collection])
        response = await model.run(query=request.message, results=results)
        logger.debug("For session: %s, model responded : %s", request.conv_id, str(response))
        # serialise latest momory and chat store objects and store them per session to update session variables
        re_serialized_memory= pickle.dumps(model.memory)
        await redis.hset(f"session:{request.conv_id}", "memory", re_serialized_memory)
        # persist updated chatstore with latest user and bot message
        chat_store_path= await redis.get("chat_store_path")
        chat_store_path= chat_store_path.decode('utf-8')
        deserialized_chat_store= deserialize(re_serialized_memory)
        deserialized_chat_store= deserialized_chat_store.chat_store
        deserialized_chat_store.persist(persist_path=chat_store_path.format(user= user_name_current, conv_id= request.conv_id))
        try: 
            final_response, match= cite(response, top_k=top_k, conv_id=request.conv_id, reranked_list=results)
        except Exception as e: 
            logger.error("Error while creating citations for session: %s. Error: : %s", request.conv_id, str(e))
            match=[]
            final_response= response.response
        try:
            # log current retrieval into csv for observability
            await log_retrievals(retrievals= retrievals, question=request.message, user= user_name_current, session_id=request.conv_id, collection_name= collection, LLM_response=response.response, citations= match, reranked_results=results)
        except Exception as e:
            logger.error(f"error on retrieval logging: {str(e)} for session: {request.conv_id}",exc_info=True)
            pass
        return final_response
    except httpx.ConnectError as e:
        logger.error(str(e)+ f" for session: {request.conv_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama, make sure it is running in the background: {str(e)}")
    except Exception as e:
        logger.error(str(e)+ f" for session: {request.conv_id}", exc_info=True)
        raise Exception(f"Error: {str(e)}, please try again later")


@app.post("/log_feedback/")
async def log_feedback(request: feedback_model, redis: Redis = Depends(lambda: app.state.redis)):
    """ Log feedback for each LLM response in the correponding user log excel

    Args:
        session_id_user (str): contains conv id and username merged
        request (log_feedback): payload model containing feedback and feedback comment
        redis: Redis db. Defaults to Depends(lambda: app.state.redis).
    """
    conv_id= request.conv_id
    user_name_current= request.username
    logger.info("Logging user: %s feedback for session: %s", user_name_current, conv_id)
    retrieval_path= RETRIEVAL_LOG_PATH.format(user= user_name_current)
    # filter citation from llm response to match it with the LLM response in the logs
    frontend_LLM_response= request.LLM_response.split(citation_header)[0]
    if os.path.exists(retrieval_path):
        try: 
            df= pd.read_excel(retrieval_path, dtype={"Feedback (like/dislike)": "string", "Feedback (Comments)": "string"})
            df.loc[(df['chat_session'] == conv_id) & (df['LLM_response'] == frontend_LLM_response), "Feedback (like/dislike)"] = request.feedback
            df.loc[(df['chat_session'] == conv_id) & (df['LLM_response'] == frontend_LLM_response), "Feedback (Comments)"] = request.feedback_comment
            df.to_excel(retrieval_path, index=False)
        except Exception as e: 
            logger.error("Error: %s while logging feedback for session: %s", str(e), conv_id)
            return False
    else: 
        return False
    return True


@app.get("/get_existing_conv_ids/{session_id_user}")
async def get_existing_conv_id(session_id_user: str, redis: Redis = Depends(lambda: app.state.redis)) -> list:
    """ Return existing conversation ids after checking if they exist in chat history
    """
    new_conv_id= session_id_user.split("$")[0]
    user= session_id_user.split("$")[1]
    init_conversation_list= await redis.get("conversations")
    init_conversation_list= json.loads(init_conversation_list)
    if new_conv_id not in init_conversation_list[user]:
        logger.error(f"Conversation not found for session: {new_conv_id}", exc_info=True)
        raise HTTPException(status_code=404, detail="Conversation not found")
    logger.info("Getting existing conversation ids for user: %s on user login", user)
    user_conv_id=[]
    user_conv_id=init_conversation_list[user]
    if not user_conv_id:
        return []
    db_details=await redis.get("chat_store_path")
    db_details= db_details.decode('utf-8')
    final_user_conv_id= check_chat_history_db(db=db_details, user=user, user_conv_id=user_conv_id, new_conv_id=new_conv_id)
    if not final_user_conv_id:
        return []
    return final_user_conv_id

@app.get("/get_user_available_docs_check_admin/{session_id_user}")
async def get_docs(session_id_user: str, redis: Redis = Depends(lambda: app.state.redis)):
    """ For a session, user, collection: get all the documents present in the collection 
    Also return True if the user has admin access. 

    Args:
        session_id_user (str): contains conv id and username merged
        redis: Redis db. Defaults to Depends(lambda: app.state.redis).
    """
    conv_id= session_id_user.split("$")[0]
    user= session_id_user.split("$")[1]
    read_collection = await redis.hget(f"session:{conv_id}", "read_collection")
    read_collection = read_collection.decode('utf-8')
    logger.info(" Getting available documents to chat for session: %s and collection %s", conv_id, read_collection)
    doc_in_collection = await get_doc_in_collection(read_collection=read_collection)
    admin_access = await check_admin(user)
    col_type= collection_type[read_collection]
    return doc_in_collection, admin_access, col_type

@app.post("/get_conversation/{session_id_user}")
async def get_conversation(session_id_user: str, request: change_session, redis: Redis = Depends(lambda: app.state.redis)):
    """ First retrieves chat history for the selected conv id and then changes redis key name from old conv id to selected conv id

    Args:
        session_id_user (str): contains new conv id and username merged
        request (change_session): Request with params: old conv_id, username, new_conv_id
        redis: Redis db. Defaults to Depends(lambda: app.state.redis).

    Raises:
        HTTPException: Check if user is in session tracker
        HTTPException: Check if session exists in session tracker

    Returns:
        chat store as json
    """
    new_conv_id= request.new_conv_id
    user_name_current= request.username
    old_conv_id= request.old_conv_id
    logger.info("User %s toggled to session id: %s", user_name_current, new_conv_id)
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)
    chat_store_path= await redis.get("chat_store_path")
    chat_store_path= chat_store_path.decode('utf-8')
    if user_name_current not in conversations_dict:
        logger.error("404 ERROR: User %s not found. Error in logic", user_name_current)
        raise HTTPException(status_code=404, detail="User not found")
    # keeping track that the current conv_id is being updated in session tracker
    if new_conv_id not in conversations_dict[user_name_current]:
        logger.error("404 ERROR: selected session %s not found", new_conv_id)
        raise HTTPException(status_code=404, detail="Conversation not found")
    key_redis_flag= await redis.exists(f"session:{old_conv_id}")
    if key_redis_flag==0:
        logger.error("404 ERROR: existing session %s not found in redis", old_conv_id)
        raise HTTPException(status_code=404, detail="Existing session not found")
    chat_store= SimpleChatStore.from_persist_path(persist_path=chat_store_path.format(user= user_name_current, conv_id= new_conv_id))
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=14000,
        chat_store=chat_store,
        chat_store_key=request.new_conv_id,
    )
    serialized_memory = pickle.dumps(memory)
    # update redis session state key with the session_id in request header
    # TODO possibly only allow toggling conv ids which are related to the current collection
    await redis.renamenx(f"session:{old_conv_id}", f"session:{new_conv_id}")
    await redis.hset(f"session:{new_conv_id}", "memory", serialized_memory)
    return chat_store.json()

@app.post("/ingest_doc_frontend/{session_id_user}/file_name")
async def ingest_file_frontend(session_id_user: str, request: Ingest_req, redis: Redis = Depends(lambda: app.state.redis)):
    user_name = session_id_user.split("$")[1]
    logger.info("User: %s ingesting file: %s to collection: %s", user_name, request.file, request.ingest_collection)
    await redis.hset(f"session:{request.conv_id}", "ingest_collection", request.ingest_collection)
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)

    milvus_username= await redis.hget(f"session:{request.conv_id}", "user_id")
    milvus_password_p2= await redis.hget(f"session:{request.conv_id}", "password")
    if not milvus_username or not milvus_password_p2:
        logger.error(f"Invalid or expired session: {request.conv_id}")
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    milvus_username= milvus_username.decode('utf-8')
    milvus_password_p2= milvus_password_p2.decode('utf-8')
    milvus_password= f"{milvus_username}:{milvus_password_p2}"
    if request.conv_id not in conversations_dict[user_name]:
        logger.error("Conversation not found for session: %s", request.conv_id)
        raise HTTPException(status_code=404, detail="Conversation not found")
    is_valid, message = validator.validate_file(request.file)
    if not is_valid:
        logger.error("Invalid file size for session: %s",request.conv_id)
        return is_valid, message
    
    # copy file to local file database
    if not os.path.exists(FILES_DB):
        os.mkdir(FILES_DB)
    shutil.copy(request.file, FILES_DB)
    logger.info("New file in %s folder: %s", FILES_DB, request.file.split("/")[-1])

    new_file= FILES_DB + "/" +  request.file.split("/")[-1]
    try:
        parsing_obj= Docling_parser()
        parsed_doc= parsing_obj.docling_ingest(file=new_file, collection_name=request.ingest_collection)
        response, message= await ingest(parsed_doc=parsed_doc, file=new_file, user_name=user_name, ingest_collection=request.ingest_collection, user_milvus_pass=milvus_password, conv_id=request.conv_id)
    except Exception as e: 
        response=False
        message=f"Ingestion unsuccesful: {str(e)} "
    return response, message

'''

@app.post("/change_collection/{session_id_user}")
async def change_collection(session_id_user:str, request: change_col, redis: Redis = Depends(lambda: app.state.redis)) -> dict:
    """ FOR CLI, change read collection to the specified collection """
    conv_id= session_id_user.split("$")[0]
    user_name_current= session_id_user.split("$")[1]
    logger.info(" Changing collection name for session: %s to: %s", conv_id, request.read_collection_name)
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)
    if request.conv_id not in conversations_dict[user_name_current]:
        logger.error(f"Conversation not found for session: {request.conv_id}", exc_info=True)
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    collection= await redis.hget(f"session:{request.conv_id}", "read_collection")
    if not collection: 
        logger.error(f"Invalid or expired session: {request.conv_id}")
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    if collection !=request.read_collection_name:
        await redis.hset(f"session:{request.conv_id}", "read_collection", request.read_collection_name)
        return {"message": f"For Conversation {request.conv_id}, current collection is {request.read_collection_name}"}
    else:
        return {"message": f"For Conversation {request.conv_id}, collection already set to {collection}"}

@app.post("/conversation/{session_id_user}/file_name")
async def ingest_file(session_id_user:str, request: Ingest_req, redis: Redis = Depends(lambda: app.state.redis)) -> tuple:
    """ Parse document through docling parser after checking file size compatibility. 
        After parsing and postprocessing, the function chunks and embeds the document into milvus (for now, with allminilm). 
        Sends a tuple as response, first element being boolean value whether ingestion was successful
        second element being the message accompanying the boolean value. 
    """
    logger.info(" Ingesting file %s for session: %s to collection: %s", request.file, request.conv_id, request.ingest_collection)
    user_name = session_id_user.split("$")[1]
    await redis.hset(f"session:{request.conv_id}", "ingest_collection", request.ingest_collection)
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)

    milvus_username= await redis.hget(f"session:{request.conv_id}", "user_id")
    milvus_password_p2= await redis.hget(f"session:{request.conv_id}", "password")
    if not milvus_username or not milvus_password_p2:
        logger.error(f"Invalid or expired session: {request.conv_id}")
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    milvus_username= milvus_username.decode('utf-8')
    milvus_password_p2= milvus_password_p2.decode('utf-8')
    milvus_password= f"{milvus_username}:{milvus_password_p2}"
    if request.conv_id not in conversations_dict[user_name]:
        logger.error("Conversation not found for session: %s", request.conv_id)
        raise HTTPException(status_code=404, detail="Conversation not found")
    # check user privileges
    client = MilvusClient(
    uri=MILVUS_URI,
    token=TOKEN)
    if MILVUS_ROOT_ROLE not in client.describe_user(user_name=milvus_username)["roles"]:
        logger.error("Unauthorized ingestion attempted for session: %s and user: %s for collection: %s", request.conv_id, user_name, request.ingest_collection)
        raise HTTPException(status_code=401, detail="You are not authorized to ingest documents")
    # check file size
    is_valid, message = validator.validate_file(request.file)
    if not is_valid:
        logger.error("Invalid file size for session: %s",request.conv_id)
        return is_valid, message
    # parse document using docling
    try:
        parsing_obj= Docling_parser()
        parsed_doc= parsing_obj.docling_ingest(file=request.file, collection_name=request.ingest_collection)
        response, message= await ingest(parsed_doc=parsed_doc, file=request.file, user_name=user_name, ingest_collection=request.ingest_collection, user_milvus_pass=milvus_password, conv_id=request.conv_id)
    except Exception as e: 
        response=False
        message=f"Ingestion unsuccesful: {str(e)} "
    return response, message
'''
    
@app.post("/create_collection/{collection_name}")
async def create_collection(collection_name: str, redis: Redis = Depends(lambda: app.state.redis)):
    """ Create an empty collection if not found in milvus

    Args:
        collection_name (str): name of collection to create
        redis: Defaults to Depends(lambda: app.state.redis).

    Returns:
        bool: if collection has been created, return True
    """
    model= col_mod[collection_name]
    dim= dim_mod[model]
    milvus_client = MilvusClient(uri=MILVUS_URI,token=TOKEN)
    milvus_client= await ingest2milvus.create_new_collection(milvus_client=milvus_client, dim=dim, collection_name=collection_name)
    if collection_name in milvus_client.list_collections(): 
        response= True
    else: 
        response= False
    milvus_client.close()
    return response

@app.get("/internal_get_vectordb/{collection}")
async def get_vector_db(collection):
    # add role based access
    return await milvus_db_as_excel(collection)


@app.post("/logout")
async def logout(request: Logout_req,redis: Redis = Depends(lambda: app.state.redis)):
    """ First removes session id from redis hash and then:
        	1. Removes jsons with empty chats
            2. Removes session ids from conversation tracker for the empty OR non existent chats 

    Args:
        request (Logout_req): request data with current conv id and username
        redis: Redis db. Defaults to Depends(lambda: app.state.redis).

    Returns:
        dict: message indicating successful logout (useful for CLI)
    """
    conv_id=request.conv_id
    username=request.user
    logger.info("Logout initiated for user id: %s, session id: %s", username, conv_id)
    # delete session key from redis on logout
    session_data = await redis.hgetall(f"session:{conv_id}")
    if session_data:
        await redis.delete(f"session:{conv_id}")
    # if session chat store is empty, remove from conversations and delete chat store entry
    chat_store_path= await redis.get("chat_store_path")
    chat_store_path= chat_store_path.decode('utf-8')
    conversations= await redis.get("conversations")
    conversations_dict= json.loads(conversations)
    conversation_id_list= copy.deepcopy(conversations_dict[username])
    try:
        for id in conversation_id_list:
            chat_store= chat_store_path.format(user= username, conv_id= id)
            empty_chat= check_empty_chats(username=username, chat_store=chat_store, id=id)
            if empty_chat:
                conversations_dict[username].remove(id)
        # persist the updated conversation tracker
        await redis.set("conversations", json.dumps(conversations_dict))
        with open(USER_HISTORY, 'w') as fp:
            json.dump(conversations_dict, fp)
    except Exception as e:
        logger.error("Error during logging out %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("Sucessful logout for user: %s", username)
    return {"message": "Logged out successfully"}