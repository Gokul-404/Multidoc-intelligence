"""
Integration tests for FastAPI endpoints in SentinelRAG.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SentinelRAG"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "timestamp" in data


def test_list_documents_default_tenant():
    response = client.get("/api/v1/documents", headers={"X-Tenant-ID": "test_tenant"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_query_validation_empty_question():
    payload = {
        "question": "",
        "document_ids": ["doc_123"],
    }
    response = client.post("/api/v1/query", json=payload, headers={"X-Tenant-ID": "test_tenant"})
    # Either validation error (422) or rejection
    assert response.status_code in [400, 422]
