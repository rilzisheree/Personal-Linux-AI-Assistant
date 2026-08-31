from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from local_ai_assistant.api import ApiServer
from local_ai_assistant.api_store import ApiStore
from local_ai_assistant.ollama import StreamEvent


class FakeOllamaClient:
    def __init__(self, _url: str, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def list_models(self) -> list[str]:
        return ["qwen3.5:4b"]

    def stream_chat(self, messages, model, cancel_event, tools, context_size):
        assert messages
        assert model == "qwen3.5:4b"
        assert not cancel_event.is_set()
        assert tools == []
        assert context_size == 8192
        yield StreamEvent(content="Hello")
        yield StreamEvent(content=" from the API")
        yield StreamEvent(done=True)


class ApiTests(unittest.TestCase):
    PASSWORD = "correct horse battery staple"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        store = ApiStore(Path(self.temp_dir.name) / "api.db", "test-session-secret-123")
        store.set_password(self.PASSWORD)
        self.server = ApiServer(
            ("127.0.0.1", 0),
            store,
            "http://ollama.test",
            "qwen3.5:4b",
            8192,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        cookie: str | None = None,
    ) -> tuple[int, dict | str, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers={
                **({"Content-Type": "application/json"} if body else {}),
                **({"Cookie": cookie} if cookie else {}),
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                raw = response.read()
                status = response.status
                headers = dict(response.headers.items())
        except HTTPError as error:
            raw = error.read()
            status = error.code
            headers = dict(error.headers.items())
        content_type = headers.get("Content-Type", "")
        result: dict | str = (
            json.loads(raw.decode("utf-8"))
            if "application/json" in content_type
            else raw.decode("utf-8")
        )
        return status, result, headers

    @staticmethod
    def session_cookie(headers: dict[str, str]) -> str:
        return headers["Set-Cookie"].split(";", 1)[0]

    def login(self) -> str:
        status, payload, headers = self.request(
            "POST",
            "/api/auth/login",
            {"password": self.PASSWORD},
        )
        self.assertEqual(status, 200)
        self.assertIsInstance(payload, dict)
        return self.session_cookie(headers)

    def test_health_password_login_and_session_lifecycle(self) -> None:
        status, payload, _ = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["ok"], True)

        cookie = self.login()
        status, payload, _ = self.request("GET", "/api/me", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(payload["user"]["id"], "local")

        status, payload, _ = self.request(
            "POST",
            "/api/auth/logout",
            cookie=cookie,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["ok"], True)
        status, payload, _ = self.request("GET", "/api/me", cookie=cookie)
        self.assertEqual(status, 401)
        self.assertIn("Authentication is required", payload["error"])

    @patch("local_ai_assistant.api.OllamaClient", FakeOllamaClient)
    def test_conversations_stream_and_persist(self) -> None:
        cookie = self.login()

        status, payload, _ = self.request(
            "POST",
            "/api/conversations",
            {"title": "Private chat"},
            cookie=cookie,
        )
        self.assertEqual(status, 201)
        conversation_id = payload["conversation"]["id"]

        status, payload, _ = self.request(
            "POST",
            f"/api/conversations/{conversation_id}/messages",
            {"content": "Say hello"},
            cookie=cookie,
        )
        self.assertEqual(status, 200)
        self.assertIn('event: token\ndata: {"content":"Hello"}', payload)
        self.assertIn('"message":"Hello from the API"', payload)

        status, payload, _ = self.request(
            "GET",
            f"/api/conversations/{conversation_id}",
            cookie=cookie,
        )
        self.assertEqual(status, 200)
        messages = payload["conversation"]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "Hello from the API")

    def test_invalid_password_and_account_registration_are_rejected(self) -> None:
        status, payload, headers = self.request(
            "POST",
            "/api/auth/login",
            {"password": "wrong-password"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "Password is incorrect.")

        status, payload, _ = self.request(
            "POST",
            "/api/auth/register",
            {"password": self.PASSWORD},
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "Endpoint not found.")

    def test_local_session_secret_is_created_when_environment_secret_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "api-session.secret"
            with patch.dict(os.environ, {"SESSION_SECRET": ""}):
                store = ApiStore(
                    Path(directory) / "api.db",
                    session_secret_path=secret_path,
                )
                store.set_password(self.PASSWORD)
                token, _ = store.create_session()
                self.assertIsNotNone(store.user_for_session(token))
            self.assertTrue(secret_path.is_file())
            self.assertGreaterEqual(len(secret_path.read_text(encoding="utf-8").strip()), 16)