def test_health(test_client):
    """ Test if the connection is stable"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}


def test_validate_true(test_client):
    """Validate that a user can obtain a JWT with correct credentials."""
    email = "test_user@example.com"
    password = "test_password"
    create_payload = {
        "email": email,
        "password": password,
        "is_admin": False,
        "rag_type": "rag",
        "workspace_ids": [],
    }
    response_create = test_client.post("/create_user", json=create_payload)
    assert response_create.status_code == 200

    payload = {
        "email": email,
        "password": password,
    }
    response = test_client.post("/validate_user", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["access_token"]
    assert body["profile"]["email"] == email


def test_validate_false(test_client):
    """For wrong credentials validation should fail."""
    email = "test_user_fail@example.com"
    password = "correct_password"
    create_payload = {
        "email": email,
        "password": password,
        "is_admin": False,
        "rag_type": "rag",
        "workspace_ids": [],
    }
    response_create = test_client.post("/create_user", json=create_payload)
    assert response_create.status_code == 200

    payload = {
        "email": email,
        "password": "wrong_password",
    }
    response = test_client.post("/validate_user", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False


def test_create_duplicate_user(test_client):
    """Creating the same user twice should return an error."""
    email = "duplicate_user@example.com"
    password = "duplicate_password"
    create_payload = {
        "email": email,
        "password": password,
        "is_admin": False,
        "rag_type": "graphrag",
        "workspace_ids": [],
    }
    response_create = test_client.post("/create_user", json=create_payload)
    assert response_create.status_code == 200

    response_duplicate = test_client.post("/create_user", json=create_payload)
    assert response_duplicate.status_code == 400
    assert "already exists" in response_duplicate.json()["detail"].lower()
