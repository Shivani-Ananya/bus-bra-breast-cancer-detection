from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_is_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "BreastSight Studio" in response.text
    assert "text/html" in response.headers["content-type"]


def test_supabase_status_is_safe_when_not_configured(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    response = client.get("/api/supabase/status")
    assert response.status_code == 200
    assert response.json()["configured"] is False
    assert "key" not in response.text.lower()


def test_cases_require_authentication():
    response = client.get("/api/cases")
    assert response.status_code == 401
