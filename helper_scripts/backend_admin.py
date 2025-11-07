from pymilvus import MilvusClient
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from app.config import MILVUS_URI, TOKEN, MILVUS_ROOT_ROLE, MILVUS_USER_ROLE, USER_DB_PATH, USER_COLLECTION_MAPPING, col_mod  # noqa: E402
from app.Ingestion_workflows.docling_parse_process import Docling_parser # noqa: E402
from pymilvus.exceptions import MilvusException  # noqa: E402
import json  # noqa: E402

def init_client():
    client = MilvusClient(
        uri=MILVUS_URI,
        token=TOKEN
        )
    return client

def create_roles():
    """ Create admin and user roles and give them the required privileges
    """
    client= init_client()
    try:
        client.create_role(role_name=MILVUS_ROOT_ROLE)
        client.create_role(role_name=MILVUS_USER_ROLE)
    except MilvusException as e:
        print(str(e))
        pass
    try:
        client.create_privilege_group(group_name='privilege_readers')
        client.add_privileges_to_group(group_name='privilege_readers', privileges=['Query', 'Search'])
    except MilvusException as e:
        print(str(e))
        pass
    try:
        client.grant_privilege_v2(role_name=MILVUS_ROOT_ROLE, privilege="CollectionAdmin", collection_name="*", db_name="default")
    except MilvusException as e:
        print(str(e))
    pass
    try:
        client.grant_privilege_v2(role_name=MILVUS_USER_ROLE, privilege="privilege_readers", collection_name="*", db_name="default")
    except MilvusException as e:
        print(str(e))
    pass
    client.close()



def make_admin(username):
    client= init_client()
    client.grant_role(user_name=username, role_name=MILVUS_ROOT_ROLE)
    client.close()
    return 

def delete_col(collection_name):
    client= init_client()
    try:
        client.drop_collection(
        collection_name=collection_name)
    except:  # noqa: E722
        pass
    client.close()
    return

def del_user_from_db(db_path_1: str, db_path_2: str, username: str):
    user_1=""
    user_2=""
    with open(db_path_1,'r+') as file:
        file_data = json.load(file)
    if file_data:
        user_1= file_data.pop(username)   
    with open(db_path_1, "w") as file:
        json.dump(file_data, file)

    with open(db_path_2,'r+') as file_2:
        file_data_2 = json.load(file_2)
    if file_data_2:
        user_2= file_data_2.pop(username)   
    with open(db_path_2, "w") as file:
        json.dump(file_data_2, file)
    return user_1, user_2

def delete_user(username):
    client= init_client()
    try:
        client.drop_user(
        user_name=username)
    except Exception as e:
        print(str(e))
        pass
    client.close()
    user_db= USER_DB_PATH
    user_collection_db= USER_COLLECTION_MAPPING
    user1, user2= del_user_from_db(db_path_1=user_db, db_path_2=user_collection_db, username=username)
    if user1 and user1==user2:
        return ("User removed:",user1)
    else:
        return ("user was not present")

def list_users():
    client= init_client()
    user_list= client.list_users()
    for user in user_list:
        print(client.describe_user(user))
        print()
    client.close()

def list_col():
    client= init_client()
    print(client.list_collections())
    client.close()

def assign_user_collection(user: str, collection: str): 
    if collection not in col_mod.keys(): 
        raise ValueError("Collection not valid")
    user_collection_db= Docling_parser.get_store(path=USER_COLLECTION_MAPPING)
    if user in user_collection_db.keys():
        user_collection_db.update({user: collection})
    else: 
        user_collection_db[user]= collection
    Docling_parser.save_object(obj=user_collection_db, file=USER_COLLECTION_MAPPING, filetype="json")
    print(user_collection_db)

if __name__ == "__main__":
    #create_roles()
    #make_admin(username= "sarva")
    #list_users()
    #delete_col(collection_name= "support_small")
    list_col()
    #delete_user(username="temp")
    #assign_user_collection(user= "user_product", collection="sales_small")
    pass
    