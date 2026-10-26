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


def test_private_image_and_report_routes_require_authentication():
    assert client.post("/api/images", json={
        "case_id": "00000000-0000-0000-0000-000000000000",
        "storage_path": "user/case/image.png",
        "original_name": "image.png",
    }).status_code == 401
    assert client.get("/api/images/example/url").status_code == 401
    assert client.get("/api/cases/example/report.pdf").status_code == 401


def test_report_pdf_builder_produces_pdf():
    from app import simple_report_pdf

    report = simple_report_pdf(["BreastSight Studio", "Research use only"])
    assert report.startswith(b"%PDF-1.4")
    assert b"Research use only" in report


def test_home_contains_persistent_workflow_controls():
    response = client.get("/")
    assert "Save case &amp; scan" not in response.text
    assert "Save case & scan" in response.text
    assert "Download PDF" in response.text
    assert "id=\"analyze-case\"" in response.text
    assert "id=\"results-dashboard\"" in response.text


def test_experiment_results_are_published_in_page_api():
    response = client.get("/api/experiment-results")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "5 experiments completed"
    assert len(data["experiments"]) == 5
    assert any(item["name"] == "Synthetic-augmented multimodal" for item in data["experiments"])
