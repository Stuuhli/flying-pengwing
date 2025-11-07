from requests.exceptions import HTTPError
import cmd
import os
import requests
import uuid
from app.auth import password_create
from app.config import API_CREATE_USER, API_VALIDATE_USER, API_CONV_START, API_CONV_SEND, API_GET_HISTORY, API_FILE_INGEST, API_LOGOUT, API_CHANGECOLLECTION, USER_DB_PATH, MILVUS_URI, MILVUS_USER_ROLE,MILVUS_ROOT_ROLE,col_mod  # noqa: F401
from app.Ingestion_workflows.milvus_RBAC import milvus_RBAC_manage
from app.utils.utils_auth import load_json
from rich import print
from colorama import Fore, Style
from pymilvus import MilvusClient
script_dir= os.path.dirname(os.path.abspath(__file__))


def show_client(username: str, password: str):
    """ Return client for user login
    """
    client = MilvusClient(
    uri=MILVUS_URI,
    token=f"{username}:{password}"
    )
    return client

def get_ingestion_collection_choice():
    """ Choose among collections in col_mod where we can ingest (should be dropped in backend first if you want entirely new collection)

    Returns:
       options[user_input] : collection_name
    """
    options={}
    for i,coll in enumerate(list(col_mod.keys()), start=1):
        options[i]=coll
    while True:
        # get collection name to ingest to
        print("Select an collection from below to ingest to:")
        for key, value in options.items():
            print(f"{key}: {value}")
        try:
            user_input = int(input("Enter the number of your choice: "))
        except Exception:
            print("Invalid choice, please try again")
            continue
        if user_input in options:
            return options[user_input]
        else:
            print("Invalid choice. Please try again.")

def get_user_collection_choice(collection_list: list):
    """ Choose among collections in current collection list for the user that we can read from. 

    Args:
        collection_list (list): list of present collections enabled for user

    Returns:
        options[user_input] : collection_name
    """
    options={}
    for i,coll in enumerate(collection_list, start=1):
        options[i]=coll
    while True:
        print("Select an collection from below to chat with:")
        for key, value in options.items():
            print(f"{key}: {value}")

        try:
            user_input = int(input("Enter the number of your choice: "))
        except Exception:
            print("Invalid choice, please try again")
            continue

        if user_input in options:
            return options[user_input]
        else:
            print("Invalid choice. Please try again.")

def init_session(sess_id, username: str, password: str, collection_name: str):
    """ Sends a post request to initialize chat memory and the session with the given session id 
    """
    data= { 
                "conv_id": str(sess_id),
                "username": username,
                "password": password,
                "read_collection_name": collection_name
               }
    try:
        response = requests.post(API_CONV_START, json=data)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return True
    except Exception as err:
        print(f"Other error occurred: {err}")
        return True
    else:
        print("Chat initialized, id: ", sess_id)
        return False

def validate_ingest_file(file_name: str):
    """ Function which checks 
    that file name exists in data folder

    Args:
        file_name (str): file to ingest
    """
    if not os.path.exists(script_dir + "/data/"+ file_name):
        return False
    return True

def check_user_DB(username: str):
    db= load_json(USER_DB_PATH)
    if username in db.keys():
        return True
    else:
        return False

class ChatCLI(cmd.Cmd):
    intro = "Please create new account by typing 'create' or login by typing 'validate <username>'\n Use ? to see list of available commands..\n"
    prompt = f"{Fore.YELLOW}User: {Style.RESET_ALL}"

    def __init__(self):
        super().__init__()
        self.authenticated = False  # Flag to track authentication status
        self.username_create = "" # username from create function
        self.session_id: uuid.UUID = ""
        self.username_current = "" # username which has been used to  login
        self.read_collection= "" # current collection to retrieve from
        self.ingest_collection= "" # current collection to ingest files into
        self.user_client= None # milvus client for user

    def do_create(self, args):
        """ Create account by entering user details, then request is sent to API to update DB with new user
        """
        if self.authenticated:
            print("The 'create' command is already disabled after successful authentication.")
            return
        self.username_create= input("Enter username: ")
        full_name = input("Enter Full name: ")
        password= input("Input password: ")
        password_re= input("Reinput password: ")
        while True: 
            admin= input("Is this new user an admin? Y/N: ")
            if admin not in ["Y", "N", "y", "n"]:
                print("Wrong input recieved")
                continue
            else:
                if admin in ["Y", "y"]:
                    admin= True
                    role= MILVUS_ROOT_ROLE
                else: 
                    admin=False
                    role= MILVUS_USER_ROLE
                break
        if password!=password_re:
            print("Password not matching, please try again!")
            return True
        elif len(password)<6:
            print("Enter password with 6 to 20 characters")
            return True
        if check_user_DB(self.username_create):
            print("Username already taken. Please use existing or choose another username")
            return True
        print("Saving user details to database")
        hashed_password= password_create(password=password_re).decode("utf-8")
        data= { 
                "username": self.username_create,
                "fullname": full_name,
                "password":hashed_password,
                "disabled": False,
                "admin": admin
               }
        try:
            response = requests.post(API_CREATE_USER, 
                                    json=data)
            response.raise_for_status()
            # create user
            user_creation= milvus_RBAC_manage()
            try:
                user_creation.create_user(username=self.username_create, password=password_re)
            except Exception:
                pass
            user_creation.assign_role(username=self.username_create, role=role)
        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        else:
            print("Details saved sucessfully!")

    def do_validate(self, username):
        """ For the username, ask password and authenticate and then start the session.
            Once session started, only chat, get_history and quit methods should be available. 
        """
        if self.authenticated:
            print("The 'validate' command is already disabled after successful authentication.")
            return
        password= input("Input password: ")
        data= {
            "username": username,
            "password": password
        }
        try:
            response = requests.post(API_VALIDATE_USER.format(user= username), 
                                    json=data)
            response.raise_for_status()
        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        else:
            print(response.json()[0])
            if response.json()[1]: 
                self.authenticated= True
            else:
                self.authenticated= False
                return True
        self.session_id = uuid.uuid4()
        if self.authenticated:
            print("Choose collection from:")
            self.user_client= show_client(username=username, password=password)
            collection_list= self.user_client.list_collections()
            if not collection_list:
                print("No collections made yet, please create through admin role")
                if MILVUS_ROOT_ROLE not in self.user_client.describe_user(user_name=username)["roles"]:
                    return True
                else: 
                    collection_name= ""
            else:
                collection_name= get_user_collection_choice(collection_list=collection_list)
        session_flag=init_session(sess_id=self.session_id, username=username, password= password, collection_name= collection_name)
        if session_flag:
            return True
        self.username_current = username
        self.read_collection= collection_name
    
    # def do_change_collection(self, collection_name):
    #     """ Api to change collection midsession

    #     Args:
    #         collection_name (str): collection name to change to
    #     """
    #     collection_list= self.user_client.list_collections()
    #     if collection_name not in collection_list:
    #         print(f"Invalid collection name, please choose from {collection_list}")
    #         return
    #     data= {
    #         "conv_id": str(self.session_id),
    #         "read_collection_name": collection_name
    #     }
    #     response = requests.post(API_CHANGECOLLECTION.format(session_id_user=str(self.session_id) + "$" + self.username_current), 
    #                              json=data)
    #     response.raise_for_status()
    #     if response.status_code == 200: 
    #         print(str(response.json()))
    #     return

    def do_chat(self, message):
        """Send a message to the chatbot
        """
        if self.session_id == "":
            print("Please authenticate first")
            return 
        data= { 
                "conv_id": str(self.session_id),
                "message":message
               }
        response = requests.post(API_CONV_SEND.format(session_id_user=str(self.session_id) + "$" + self.username_current), 
                                 json=data)
        if response.status_code == 200: 
            print(str(response.json()))
        else:
            print("Error:", response.text)

    # def do_get_history(self, arg):
    #     """ Retreive conversation history for current session """
    #     if self.session_id == "":
    #         print("Please authenticate first")
    #         return 
    #     response = requests.get(API_GET_HISTORY.format(session_id_user=str(self.session_id) + "$" + self.username_current))
    #     if response.status_code == 200:
    #         print(response.text)
    #     else:
    #         print("Error:", response.text)

    # def do_ingest(self, file_name):
    #     """ Ingest a file if present in "data" folder in the repo
    #     """
    #     if str(self.session_id) == "":
    #         print("Please authenticate first")
    #         return 
    #     if self.username_current!="sarva":
    #         print("User not allowed to ingest document")
    #         return 
    #     # check if file is pdf and present
    #     if not validate_ingest_file(file_name=file_name):
    #         print("Invalid file, please check format or location")
    #         return 
    #     # check if there is an ingest collection to ingest into
    #     if not self.ingest_collection:
    #         print("Choose collection from:")
    #         self.ingest_collection= get_ingestion_collection_choice()
    #     print("Ingesting...")
    #     full_file_name= "./data/"+ file_name
    #     data= { 
    #             "conv_id": str(self.session_id),
    #             "file":full_file_name,
    #             "ingest_collection": self.ingest_collection
    #            }
    #     response = requests.post(API_FILE_INGEST.format(session_id_user=str(self.session_id) + "$" + self.username_current), 
    #                              json=data, timeout=18000)
    #     if response.status_code == 200: 
    #         if response.json()[0]:
    #             print(response.json()[1])
    #         else:
    #             print(response.json()[1])
    #     else:
    #         print("Error:", response.text)

    def do_quit(self, arg):
        """Exit the chat
        """
        data= {
            "conv_id": str(self.session_id),
            "user": self.username_current
        }
        response= requests.post(API_LOGOUT, json=data)
        if response.status_code==200:
            print(response.json()["message"])
        else:
            print("Could not log out, please try again!")
        return True
    
    def get_names(self):
        """ If authentication done in current CLI session, creation and validation is no longer possible 
        """
        all_commands = super().get_names()
        if self.authenticated:
            # Exclude 'do_create' and 'do_validate' if authenticated
            return [cmd for cmd in all_commands if cmd not in ['do_create', 'do_validate']]
        return all_commands
    
    def help_create(self):
        if not self.authenticated:
            print("Create a new user or resource.")
        else:
            print("This command is disabled after authentication.")

    def help_validate(self):
        if not self.authenticated:
            print("Validate the user by authenticating with a database.")
        else:
            print("This command is disabled after authentication.")

if __name__ == "__main__":
    ChatCLI().cmdloop()
