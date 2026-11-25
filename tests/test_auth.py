"""Google sign-in: signed sessions, the 401 gate on patient routes, and the redirect to Google."""
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", "test-id.apps.googleusercontent.com")
    monkeypatch.setattr(s, "google_client_secret", "test-secret")
    monkeypatch.setattr(s, "public_url", "http://localhost:8000")
    monkeypatch.setattr(s, "google_redirect_uri", "")
    return TestClient(app, follow_redirects=False)


def test_session_tokens_are_signed_and_expire():
    tok = auth._sign({"email": "a@b.c", "exp": int(time.time()) + 60})
    assert auth._unsign(tok)["email"] == "a@b.c"
    assert auth._unsign(tok[:-2] + "00") is None                              # tampered
    assert auth._unsign(auth._sign({"email": "a@b.c", "exp": int(time.time()) - 1})) is None   # expired
    assert auth._unsign("garbage") is None


def test_patient_routes_need_sign_in(client):
    assert client.post("/api/consult/start?speak=false").status_code == 401
    assert client.get("/api/auth/me").json() == {"enabled": True, "user": None}


def test_signed_in_patient_gets_through(client):
    tok = auth._sign({"email": "p@x.org", "name": "Pat", "picture": "", "exp": int(time.time()) + 60})
    client.cookies.set(auth.COOKIE, tok)
    assert client.get("/api/auth/me").json()["user"]["name"] == "Pat"
    assert client.post("/api/consult/start?speak=false").status_code == 200
    client.post("/api/auth/logout")


def test_login_redirects_to_google_with_state(client):
    r = client.get("/api/auth/google/login")
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert r.status_code == 307 and q["client_id"] == ["test-id.apps.googleusercontent.com"]
    assert q["redirect_uri"] == ["http://localhost:8000/api/auth/google/callback"] and q["state"]
    assert "hva_oauth_state" in r.headers["set-cookie"]


def test_callback_rejects_a_forged_state(client):
    r = client.get("/api/auth/google/callback?code=x&state=forged")
    assert "login_error=state_mismatch" in r.headers["location"]


def test_sign_in_is_skipped_when_not_configured(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", "")
    monkeypatch.setattr(s, "local_auth", False)          # optional name + email sign-in is off by default
    c = TestClient(app)
    assert c.get("/api/auth/me").json()["enabled"] is False
    assert c.post("/api/consult/start?speak=false").status_code == 200


def _signed_in(client, sub):
    client.cookies.set(auth.COOKIE, auth._sign({"sub": sub, "email": sub + "@x.org", "name": sub, "exp": int(time.time()) + 60}))


def test_history_is_saved_listed_and_private_to_its_owner(client):
    snap = {"title": "fever and cough", "level": "see_doctor", "log": [{"role": "user", "content": "fever"}], "result": None, "progress": 0.4}
    _signed_in(client, "alice")
    assert client.put("/api/history/abc123xyz", json=snap).status_code == 200
    assert [h["title"] for h in client.get("/api/history").json()] == ["fever and cough"]
    assert client.get("/api/history/abc123xyz").json()["log"][0]["content"] == "fever"
    _signed_in(client, "bob")                                       # another patient sees nothing and cannot read or delete it
    assert client.get("/api/history").json() == []
    assert client.get("/api/history/abc123xyz").status_code == 404
    client.delete("/api/history/abc123xyz")
    _signed_in(client, "alice")
    assert client.get("/api/history/abc123xyz").status_code == 200
    client.delete("/api/history/abc123xyz")
    assert client.get("/api/history").json() == []


def test_history_needs_sign_in(client):
    assert client.get("/api/history").status_code == 401


def test_assistant_chats_are_saved_with_their_kind_and_patient(client):
    _signed_in(client, "carol")
    snap = {"kind": "assistant", "title": "warfarin and ibuprofen", "log": [{"role": "user", "content": "Does warfarin interact with ibuprofen?"}],
            "patient": {"patient_id": "P1", "full_name": "Test Patient"}}
    assert client.put("/api/history/asst00001", json=snap).status_code == 200
    assert client.put("/api/history/cons00001", json={**snap, "kind": "consult", "patient": None}).status_code == 200
    kinds = {h["id"]: h["kind"] for h in client.get("/api/history").json()}
    assert kinds == {"asst00001": "assistant", "cons00001": "consult"}
    got = client.get("/api/history/asst00001").json()
    assert got["kind"] == "assistant" and got["patient"]["patient_id"] == "P1"
    assert client.put("/api/history/bad00001", json={**snap, "kind": "other"}).status_code == 422
    for i in kinds:
        client.delete(f"/api/history/{i}")


def test_local_sign_in_for_the_lan_deployment(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", "")
    monkeypatch.setattr(s, "local_auth", True)
    c = TestClient(app)
    assert c.get("/api/auth/me").json() == {"enabled": True, "user": None}
    assert c.get("/api/auth/methods").json() == {"google": False, "google_url": None, "local": True}
    assert c.post("/api/consult/start?speak=false").status_code == 401
    assert c.post("/api/auth/local", json={"name": "Asha", "email": "not-an-email"}).status_code == 422
    assert c.post("/api/auth/local", json={"name": "Asha", "email": "Asha@Example.com"}).status_code == 200
    me = c.get("/api/auth/me").json()["user"]
    assert me["name"] == "Asha" and me["email"] == "asha@example.com"
    assert c.post("/api/consult/start?speak=false").status_code == 200


def test_local_sign_in_is_refused_when_not_enabled(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "local_auth", False)
    assert TestClient(app).post("/api/auth/local", json={"name": "A", "email": "a@b.co"}).status_code == 503


def test_google_offered_on_the_registered_public_name_only(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", "id")
    monkeypatch.setattr(s, "google_client_secret", "secret")
    monkeypatch.setattr(s, "public_url", "https://vitalis.vercel.app")
    c = TestClient(app, base_url="https://vitalis.vercel.app")
    assert c.get("/api/auth/methods").json()["google"] is True
    assert TestClient(app, base_url="https://other.example.com").get("/api/auth/methods").json()["google"] is False


def test_google_offered_on_any_host_when_public_url_is_unset(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", "id")
    monkeypatch.setattr(s, "google_client_secret", "secret")
    monkeypatch.setattr(s, "public_url", "")
    m = TestClient(app, base_url="https://preview-123.vercel.app").get("/api/auth/methods").json()
    assert m["google"] and m["google_url"] == "/api/auth/google/login"

