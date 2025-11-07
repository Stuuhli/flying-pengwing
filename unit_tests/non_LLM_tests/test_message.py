"""1. Test message: test the correct collection is returned from redis before milvus hybrid retrieve.
2. Test message: test if milvus hybrid retrieve does not work, it returns error (eg. embedding model not present)
3. Test milvus hybrid retrieve: embedding fucntion returns what is needed and the model-dim is correct (from main fucntion as well)
4. Test milvus hybrid retrieve: test what happens if k>= vector k
5. Test milvus hybrid retrieve: make sure async milvus client is called with correct uri and token
6. Test milvus hybrid retrieve: make sure async milvus client is closed after retrieval
7. Test milvus hybrid retrieve: mock rerank fucntion and make sure its called with correct params
8. Test milvus hybrid retrieve: make sure func returns 2 outputs (non empty)
9. Test milvus hybrid retrieve: test what happens if encode text or milvus client raises an exception
"""
import pytest
import json
import pickle
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from app.main import add_message
from app.config import MILVUS_URI, TOKEN, col_mod, topk_mod

@pytest.mark.asyncio
async def test_add_message_invalid_conversation():
    # Setup
    session_id_user = "invalid$user1"
    request = MagicMock(conv_id="invalid", message="Test")
    redis_mock = AsyncMock()
    
    # Return conversations that don't include the requested ID
    redis_mock.get.return_value = '{"user1": ["conv123"]}'
    
    # Test error handling
    with pytest.raises(HTTPException) as excinfo:
        await add_message(session_id_user, request, redis_mock)
    
    assert excinfo.value.status_code == 404