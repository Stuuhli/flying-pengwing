from app.auth import password_create
from app.utils.utils_auth import write_json, load_json
from unit_tests.non_LLM_tests.conftest import TEST_DB_PATH
from pathlib import Path

def test_health(test_client):
    """ Test if the connection is stable
    """
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}

def test_validate_true(test_client, make_db):
    """ For the correct login details, the response json should have True
    """
    payload= {
        "username": make_db[list(make_db.keys())[0]]["username"],
        "password": "test",
        "USER_DB_PATH": TEST_DB_PATH

    }
    response = test_client.post(f"/validate_user/{list(make_db.keys())[0]}", json= payload)
    assert response.status_code == 200
    assert response.json()[1] == True

def test_validate_false(test_client, make_db):
    """ For the wrong login details, the response json should have False
    """
    payload= {
        "username": make_db[list(make_db.keys())[0]]["username"],
        "password": "test_wrong",
        "USER_DB_PATH": TEST_DB_PATH

    }
    response = test_client.post(f"/validate_user/{list(make_db.keys())[0]}", json= payload)
    assert response.status_code == 200
    assert response.json()[1] == False

def test_create_new_user(test_client):
    """ send payload and check userdb for the exact payload with password match
    """
    payload_create= {
        "username": "new_user",
        "full_name": "new user",
        "hashed_password": password_create("new_test").decode("utf-8"),
        "disabled": False
    }
    write_json(username=payload_create["username"], new_data=payload_create, filename=TEST_DB_PATH)
    payload_valid= {
        "username": payload_create["username"],
        "password": "new_test",
        "USER_DB_PATH": TEST_DB_PATH
    }
    response = test_client.post(f"/validate_user/{payload_create['username']}", json= payload_valid)
    Path(TEST_DB_PATH).unlink()
    assert response.status_code == 200
    assert response.json()[1] == True
