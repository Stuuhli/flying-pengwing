import os
import sys
script_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(script_dir)

from app.auth import password_create
from app.main import app
from app.utils.utils_LLM import milvus_hybrid_retrieve
from app.utils.utils_auth import write_json, load_json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import json
import pytest
import test
import uuid
import pytest
import pytest_asyncio
from redis import asyncio as redis


# Mock data for user database
mock_user_db = {
    "test_user": {
        "username": "test_user",
        "full_name": "Test User",
        "hashed_password": password_create("test").decode("utf-8"),
        "disabled": False
    }
}
TEST_DB_PATH = "unit_tests/non_LLM_tests/test_auth.json"



@pytest.fixture(scope="function")
def make_db():
    """ To test existing db and testing validate feature. Create a db in path and then tear it down when done """
    with open(TEST_DB_PATH, 'w') as file:
        file_data= mock_user_db
        json.dump(file_data, file, indent = 4)
    
    with open(TEST_DB_PATH, 'r+') as file:
        file_data = json.load(file)
        yield file_data
    # Cleanup: Remove the file after the test
    if os.path.exists(os.path.join(os.getcwd(), TEST_DB_PATH)):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="function")
def test_client():
    """Create a test client that uses the override_get_db fixture to return a session."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def db_files(tmp_path):
    """ To create a temp folder which holds the db files of user_history and chat memory """
    # Create a subdirectory for API-generated files
    api_dir = tmp_path / "test_DBs"
    api_dir.mkdir()
    
    # Yield the directory path for use in tests
    yield api_dir

    #Teardown: Remove all files in the directory
    for file in api_dir.iterdir():
        file.unlink()
    
    #Remove the directory itself
    api_dir.rmdir()

# Fixture to generate a random user id
@pytest.fixture()
def user_id() -> uuid.UUID:
    """Generate a random user id."""
    return str(uuid.uuid4())

# Fixture to generate a user payload
@pytest.fixture()
def user_payload_init(user_id):
    """Generate a user payload."""
    return {
        "conv_id": user_id, 
        "username": "test_username",
        "password": "test_password",
        "read_collection_name": "test_col"
    }

# Fixture to generate a user payload
@pytest.fixture()
def user_payload_message_wrong(user_payload_init):
    """Generate a user payload."""
    return {
        "conv_id": user_payload_init["conv_id"] + "invalid", 
        "message": "This is user prompt"
    }

# Fixture to generate a user payload
@pytest.fixture()
def user_payload_message(user_payload_init):
    """Generate a user payload."""
    return {
        "conv_id": user_payload_init["conv_id"], 
        "message": "This is user prompt"
    }


    
