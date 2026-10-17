# 🚗 Used Car Price Prediction

A Flask web app that predicts the resale price of a used car using a **Linear Regression** model (scikit-learn).

## Project Structure

```
├── app.py              # Flask app (web form + JSON API)
├── train_model.py      # Builds dataset, trains model, saves model.pkl
├── model.pkl           # Trained pipeline (preprocessing + LinearRegression)
├── requirements.txt
├── .gitignore
├── Dockerfile
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── tests/
    └── test_app.py
```

## Features used for prediction

| Feature | Description |
|---|---|
| Year of purchase | Converted to car age |
| Kilometres driven | Odometer reading |
| Engine capacity (cc) | Engine size |
| Previous owners | 1–6 |
| Fuel type | Petrol / Diesel / CNG |
| Transmission | Manual / Automatic |

Numeric features are standardised and categorical features one-hot encoded, then fed to `LinearRegression`. Prices are in **lakhs of INR**.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_model.py            # creates model.pkl (prints R² and MAE)
python app.py                    # open http://localhost:5000
```

## API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"year":2019,"kms_driven":45000,"engine_cc":1200,"owners":1,"fuel_type":"Petrol","transmission":"Manual"}'
```

Response:

```json
{"price_lakhs": 4.87, "price_inr": 487000}
```

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t car-price-predictor .
docker run -p 5000:5000 car-price-predictor
```

## Using real data

`train_model.py` generates a synthetic dataset so the project runs out of the box. To use a real dataset (e.g. from Kaggle's CarDekho data), replace `load_data()` with `pd.read_csv("cars.csv")` and make sure it has the columns `car_age, kms_driven, engine_cc, owners, fuel_type, transmission, price_lakhs`. Then re-run `python train_model.py`.

> Note: predictions from the bundled model are for demonstration and not real market valuations.
