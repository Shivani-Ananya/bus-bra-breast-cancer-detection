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
