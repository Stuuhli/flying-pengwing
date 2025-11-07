from app.config import USER_DB_PATH, BACKEND_FASTAPI_LOG
from pydantic import BaseModel
import json
import os
from app.utils.utils_logging import initialize_logging, logger
initialize_logging(BACKEND_FASTAPI_LOG)

class user_auth_format(BaseModel):
    """ Request template for sending data for creation of new user account """
    username: str
    fullname: str
    password: str
    disabled: bool = False
    admin: bool = False

class user_auth_validate(BaseModel):
    """ Request template for sending data for validation of user account """
    username: str
    password: str
    USER_DB_PATH: str = USER_DB_PATH

def write_json(username, new_data, filename=USER_DB_PATH):
    """ Write user data into existing db for the new user name 
    """
    empty_dict= {}
    # If the json database does not exist
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(str(empty_dict))
    # Open the file 
    with open(filename,'r+') as file:
        try:
            file_data = json.load(file)
        except Exception as e: 
            logger.error("Error occured: %s", str(e))
            with open(filename, 'w') as file:
                file.write(str(empty_dict))
            file_data = json.load(file)
        file_data[username]= new_data
        file.seek(0)
        json.dump(file_data, file, indent = 4)


def load_json(filename=USER_DB_PATH):
    """ Load the user db for authentication 
    """
    empty_dict= {}
    # If the json database does not exist
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(str(empty_dict))
    # Open the file 
    with open(filename,'r+') as file:
        try:
            file_data = json.load(file)
        except Exception as e: 
            logger.error("Error occured: %s", str(e))
            with open(filename, 'w') as file:
                file.write(str(empty_dict))
            file_data = json.load(file)
    return file_data
