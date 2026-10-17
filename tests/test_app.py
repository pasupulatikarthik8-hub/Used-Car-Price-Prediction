import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


VALID = {
    "year": 2019,
    "kms_driven": 45000,
    "engine_cc": 1200,
    "owners": 1,
    "fuel_type": "Petrol",
    "transmission": "Manual",
}


def test_home_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Used Car Price Predictor" in res.data


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_form_prediction(client):
    res = client.post("/", data=VALID)
    assert res.status_code == 200
    assert b"Estimated price" in res.data


def test_api_prediction(client):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert body["price_lakhs"] > 0
    assert body["price_inr"] > 0


def test_newer_car_costs_more_than_older(client):
    new = client.post("/api/predict", json={**VALID, "year": 2025, "kms_driven": 5000}).get_json()
    old = client.post("/api/predict", json={**VALID, "year": 2011, "kms_driven": 150000}).get_json()
    assert new["price_lakhs"] > old["price_lakhs"]


def test_api_rejects_missing_fields(client):
    res = client.post("/api/predict", json={"year": 2019})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_api_rejects_invalid_year(client):
    res = client.post("/api/predict", json={**VALID, "year": 1800})
    assert res.status_code == 400


def test_form_shows_error_on_bad_input(client):
    res = client.post("/", data={**VALID, "fuel_type": "Steam"})
    assert res.status_code == 200
    assert b"Invalid fuel type" in res.data
