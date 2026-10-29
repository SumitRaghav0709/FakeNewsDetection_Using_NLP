import pytest
from fastapi.testclient import TestClient
from fake_news_app import app, clean_text, MAX_INPUT_CHARS

client = TestClient(app)

def test_clean_text():
    raw_html = "<p>Breaking news! Visit https://example.com for more info.</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "URL" in cleaned
    assert "Breaking news!" in cleaned

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Fake News Detection" in response.text
    assert "Hugging Face Transformer" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_directory_exists"] is True
    assert "cpu" in data["device"] or "cuda" in data["device"]

def test_predict_real_news():
    payload = {
        "headline": "Federal Reserve Holds Interest Rates Steady Amid Inflation Data",
        "body_text": "WASHINGTON (Reuters) - The Federal Reserve held interest rates unchanged on Wednesday and signaled borrowing costs are likely to remain on hold until central bankers gain greater confidence that inflation is moving sustainably toward their target."
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["REAL", "FAKE"]
    assert 0.0 <= data["p_fake"] <= 1.0
    assert 0.0 <= data["p_real"] <= 1.0
    assert abs((data["p_fake"] + data["p_real"]) - 1.0) < 1e-4
    assert "notice" in data
    print("\n[Real News Test Output]:", data)

def test_predict_fake_news():
    payload = {
        "headline": "SHOCKING SECRET: Aliens Caught Running Secret Government Base Under Antarctica!",
        "body_text": "An anonymous whistleblower from an undisclosed agency revealed proof that shape-shifting reptilian extraterrestrials have taken full control of world governments from a secret bunker deep under the South Pole."
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["REAL", "FAKE"]
    assert 0.0 <= data["p_fake"] <= 1.0
    assert 0.0 <= data["p_real"] <= 1.0
    print("\n[Fake News Test Output]:", data)

def test_predict_empty_input():
    payload = {
        "headline": "",
        "body_text": ""
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422
    assert "Enter a headline or article body" in response.json()["detail"]

def test_predict_oversized_input():
    payload = {
        "headline": "A" * 100,
        "body_text": "B" * (MAX_INPUT_CHARS + 500)
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code in [413, 422]

if __name__ == "__main__":
    print("Running Fake News Detection Test Suite...")
    test_clean_text()
    print("[PASS] test_clean_text passed")
    test_home_page()
    print("[PASS] test_home_page passed")
    test_health_check()
    print("[PASS] test_health_check passed")
    test_predict_real_news()
    print("[PASS] test_predict_real_news passed")
    test_predict_fake_news()
    print("[PASS] test_predict_fake_news passed")
    test_predict_empty_input()
    print("[PASS] test_predict_empty_input passed")
    test_predict_oversized_input()
    print("[PASS] test_predict_oversized_input passed")
    print("\nAll 7 test cases successfully verified!")
