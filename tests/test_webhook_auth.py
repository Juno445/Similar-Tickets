import hashlib
import hmac
import json
import os
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import types

sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=lambda name: None))
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('pyplot'))
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

import src.webhook_server as server
from src.webhook_server import app

client = TestClient(app)

def create_signature(payload: str, secret: str) -> str:
    """Create HMAC-SHA256 signature for webhook payload."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


class TestWebhookAuthentication:
    """Test webhook HMAC authentication."""

    def setup_method(self):
        """Set up test fixtures."""
        self.payload = '{"ticket_id": 12345}'
        self.secret = "test-webhook-secret"
        self.valid_signature = create_signature(self.payload, self.secret)

    def test_webhook_without_secret_env_allows_all(self, monkeypatch):
        """When WEBHOOK_SECRET is not set, all requests should be allowed."""
        monkeypatch.setenv("WEBHOOK_SECRET", "")
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        
        # Mock the processing function
        mock_process = MagicMock(return_value=True)
        monkeypatch.setattr(server, "_process_sync", mock_process)
        
        response = client.post(
            "/webhook",
            data=self.payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        assert response.json() == {"merged": True}

    def test_webhook_with_valid_signature(self, monkeypatch):
        """Valid signature should allow request through."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        # Mock the processing function
        mock_process = MagicMock(return_value=True)
        monkeypatch.setattr(server, "_process_sync", mock_process)
        
        response = client.post(
            "/webhook",
            data=self.payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": self.valid_signature
            }
        )
        
        assert response.status_code == 200
        assert response.json() == {"merged": True}

    def test_webhook_with_invalid_signature(self, monkeypatch):
        """Invalid signature should return 401."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        response = client.post(
            "/webhook",
            data=self.payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": "invalid-signature"
            }
        )
        
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid signature"}

    def test_webhook_missing_signature_header(self, monkeypatch):
        """Missing signature header should return 401 when secret is configured."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        response = client.post(
            "/webhook",
            data=self.payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 401
        assert response.json() == {"detail": "missing signature header"}

    def test_webhook_with_malformed_json(self, monkeypatch):
        """Malformed JSON should return 400."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        malformed_payload = '{"ticket_id": invalid}'
        signature = create_signature(malformed_payload, self.secret)
        
        response = client.post(
            "/webhook",
            data=malformed_payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature
            }
        )
        
        assert response.status_code == 400
        assert response.json() == {"detail": "invalid payload"}

    def test_signature_timing_attack_protection(self, monkeypatch):
        """Test that signature comparison is timing-attack safe."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        # This test ensures we're using hmac.compare_digest
        # which is timing-attack safe
        with patch('hmac.compare_digest') as mock_compare:
            mock_compare.return_value = False
            
            response = client.post(
                "/webhook",
                data=self.payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": "some-signature"
                }
            )
            
            assert response.status_code == 401
            mock_compare.assert_called_once()

    def test_different_payload_different_signature(self, monkeypatch):
        """Different payload should require different signature."""
        monkeypatch.setenv("WEBHOOK_SECRET", self.secret)
        
        different_payload = '{"ticket_id": 99999}'
        
        # Use signature for original payload with different payload
        response = client.post(
            "/webhook",
            data=different_payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": self.valid_signature
            }
        )
        
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid signature"} 