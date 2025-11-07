from pydantic import BaseModel

class session_start_req(BaseModel):
    """ Request template for starting a new session and initializing components like chat store and chat db"""
    conv_id: str
    username: str
    password: str

class feedback_model(BaseModel):
    conv_id: str
    username: str
    LLM_response: str
    feedback: str
    feedback_comment: str

class Message_request(BaseModel):
    """ Message template for data sent to API """
    conv_id: str
    message: str

class Ingest_req(BaseModel):
    """ Message template for data sent to API """
    conv_id: str
    file: str
    ingest_collection: str

class change_session(BaseModel):
    """ When user toggles between session ids """
    old_conv_id: str
    new_conv_id: str
    username: str

class change_col(BaseModel):
    """ Change collection to read from """
    conv_id: str
    read_collection_name: str

class Logout_req(BaseModel):
    """ Logout template """
    conv_id: str
    user: str

class RerankResult(BaseModel):
    """ Reranked results template mirroring pymilvus reranker"""
    text:str
    score:float
    index:int