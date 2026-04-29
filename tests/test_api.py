"""
API Tests - FastAPI endpoint testleri
"""
import pytest
from unittest.mock import patch, MagicMock
import os

# Test öncesi mock API key
os.environ["GROQ_API_KEY"] = "test_key_for_testing"


@pytest.fixture
def client():
    """FastAPI test client - yeni httpx uyumlu"""
    from fastapi.testclient import TestClient
    
    with patch('groq.Groq') as mock_groq:
        mock_groq.return_value = MagicMock()
        from main import app
        # Yeni httpx için transport parametresi kullanılıyor
        test_client = TestClient(app=app)
        yield test_client


class TestHealthEndpoint:
    """Health check endpoint testleri"""

    def test_health_returns_status(self, client):
        """Health endpoint dönüş yapmalı"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "files" in data


class TestUploadEndpoint:
    """Upload endpoint testleri"""

    def test_upload_rejects_unsupported_format(self, client):
        """Desteklenmeyen format reddedilmeli"""
        files = {"files": ("test.xyz", b"content", "application/octet-stream")}
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["status"] == "error"


class TestChatEndpoint:
    """Chat endpoint testleri"""

    def test_chat_without_documents_returns_error(self, client):
        """Doküman yokken hata dönmeli"""
        from main import state
        state["index"] = None
        state["metadata"] = []
        
        response = client.post(
            "/chat/stream",
            json={"message": "test", "show_sources": True, "active_files": []}
        )
        assert response.status_code == 400