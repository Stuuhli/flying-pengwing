import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from redis import asyncio as redis

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.append(str(repo_root))

TEST_DB_PATH = script_dir / "test_app.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

from app.main import app  # noqa: E402
from app.utils.utils_LLM import milvus_hybrid_retrieve  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def cleanup_database():
    yield
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture(scope="function")
def test_client():
    """Create a test client for the FastAPI application."""
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
def user_id() -> str:
    """Generate a random user id."""
    import uuid

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
