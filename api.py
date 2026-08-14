import os
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from prometheus_fastapi_instrumentator import Instrumentator


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction using MLflow and Gradient Boosting",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

# --------------------------------------------------
# Prometheus monitoring
# --------------------------------------------------

Instrumentator().instrument(app).expose(app)

# --------------------------------------------------
# Input schema
#
# --------------------------------------------------


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Feature definitions
# --------------------------------------------------

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# --------------------------------------------------
# Create preprocessing object
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# Fit preprocessing using original dataset
# --------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "ML DATASET_Telco-Customer-Churn.csv",
)

print("Initializing API preprocessing...")

try:

    original_data = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    original_data["TotalCharges"] = pd.to_numeric(
        original_data["TotalCharges"],
        errors="coerce",
    )

    # Same treatment used during preprocessing
    original_data["TotalCharges"] = (
        original_data["TotalCharges"].fillna(0)
    )

    # Remove target and customer ID
    X_original = original_data.drop(
        columns=["Churn", "customerID"]
    )

    # Fit ONLY using the original training-style features
    preprocessor.fit(X_original)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print("API preprocessing initialized.")
    print(
        "Number of processed features:",
        len(feature_names),
    )

except Exception as e:

    print(
        "ERROR: Could not initialize preprocessing."
    )
    print("Error:", e)

    raise RuntimeError(
        "API preprocessing initialization failed."
    )


# --------------------------------------------------
# Load Gradient Boosting model
# --------------------------------------------------

if os.path.exists("/app/model"):
    MODEL_PATH = "/app/model"
else:
    MODEL_PATH = (
        "mlruns/1/models/"
        "m-70925c8db41c42dd967c6478ffa725fb/"
        "artifacts"
    )

print("Loading production model...")

try:

    # Load using sklearn flavor.
    # This gives access to predict_proba().
    model = mlflow.sklearn.load_model(
        MODEL_PATH
    )

    print(
        "Production Gradient Boosting model loaded successfully."
    )

except Exception as e:

    print(
        "ERROR: Could not load production model."
    )
    print("Error:", e)

    raise RuntimeError(
        "Production model loading failed."
    )


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "docs": "/docs",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CustomerChurnGradientBoosting",
        "features": len(
            preprocessor.get_feature_names_out()
        ),
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    # ----------------------------------------------
    # Convert request to DataFrame
    # ----------------------------------------------

    input_data = pd.DataFrame(
        [customer.model_dump()]
    )

    # ----------------------------------------------
    # Apply EXACT same preprocessing
    # ----------------------------------------------

    processed_data = preprocessor.transform(
        input_data
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        processed_data,
        columns=feature_names,
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = model.predict(
        processed_df
    )

    prediction_value = int(
        prediction[0]
    )

    # ----------------------------------------------
    # Churn label
    # ----------------------------------------------

    if prediction_value == 1:
        churn = "Yes"
    else:
        churn = "No"

    # ----------------------------------------------
    # Churn probability
    # ----------------------------------------------

    probabilities = model.predict_proba(
        processed_df
    )

    churn_probability = float(
        probabilities[0][1]
    )

    # ----------------------------------------------
    # Return response
    # ----------------------------------------------

    return {
        "prediction": prediction_value,
        "churn": churn,
        "churn_probability": round(
            churn_probability,
            4,
        ),
    }
