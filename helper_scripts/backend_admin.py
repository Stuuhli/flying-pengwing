from pymilvus import MilvusClient
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from app.config import MILVUS_URI, TOKEN, MILVUS_ROOT_ROLE, MILVUS_USER_ROLE, col_mod  # noqa: E402
from pymilvus.exceptions import MilvusException  # noqa: E402
from sqlalchemy import select  # noqa: E402
from app.db import SessionLocal, User, Workspace, Collection  # noqa: E402

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

def create_workspace(name: str, description: str = ""):
    with SessionLocal() as session:
        workspace = session.execute(select(Workspace).where(Workspace.name == name)).scalar_one_or_none()
        if workspace:
            print(f"Workspace '{name}' already exists (id={workspace.id})")
            return workspace
        workspace = Workspace(name=name, description=description)
        session.add(workspace)
        session.commit()
        session.refresh(workspace)
        print(f"Workspace '{name}' created with id={workspace.id}")
        return workspace


def create_collection(name: str, description: str = "", document_count: int = 0):
    with SessionLocal() as session:
        collection = session.execute(select(Collection).where(Collection.name == name)).scalar_one_or_none()
        if collection:
            print(f"Collection '{name}' already exists (id={collection.id})")
            return collection
        collection = Collection(name=name, description=description, document_count=document_count)
        session.add(collection)
        session.commit()
        session.refresh(collection)
        print(f"Collection '{name}' created with id={collection.id}")
        return collection

def delete_user(username):
    client= init_client()
    try:
        client.drop_user(
        user_name=username)
    except Exception as e:
        print(str(e))
        pass
    client.close()
    with SessionLocal() as session:
        user = session.execute(select(User).where(User.email == username)).scalar_one_or_none()
        if user:
            session.delete(user)
            session.commit()
            return ("User removed", username)
    return ("user was not present", username)

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

def assign_user_collection(user: str, workspace_name: str, collection: str, description: str = ""):
    if collection not in col_mod.keys():
        raise ValueError("Collection not valid")
    with SessionLocal() as session:
        user_obj = session.execute(select(User).where(User.email == user)).scalar_one_or_none()
        if not user_obj:
            raise ValueError("User not found in database")
        workspace = session.execute(select(Workspace).where(Workspace.name == workspace_name)).scalar_one_or_none()
        if not workspace:
            workspace = Workspace(name=workspace_name, description=description)
            session.add(workspace)
            session.flush()
        collection_obj = session.execute(select(Collection).where(Collection.name == collection)).scalar_one_or_none()
        if not collection_obj:
            collection_obj = Collection(name=collection, description="", document_count=0)
            session.add(collection_obj)
            session.flush()
        if collection_obj not in workspace.collections:
            workspace.collections.append(collection_obj)
        if workspace not in user_obj.workspaces:
            user_obj.workspaces.append(workspace)
        session.commit()
        print(f"Assigned {user} to workspace '{workspace.name}' with collection '{collection_obj.name}'")

if __name__ == "__main__":
    #create_roles()
    #make_admin(username= "sarva")
    #list_users()
    #delete_col(collection_name= "support_small")
    list_col()
    #delete_user(username="temp")
    #assign_user_collection(user="user_product@example.com", workspace_name="ciso-team", collection="sales_small")
    pass
    