from pymilvus import MilvusClient
from app.config import MILVUS_URI, TOKEN

class milvus_RBAC_manage():
    client = MilvusClient(
            uri=MILVUS_URI,
            token=TOKEN
        )
    # assign role to user
    @classmethod
    def assign_role(cls, username, role):
        cls.client.grant_role(user_name=username, role_name=role)
    @classmethod    
    # create user
    def create_user(cls, username, password):
        cls.client.create_user(user_name=username, password=password)

