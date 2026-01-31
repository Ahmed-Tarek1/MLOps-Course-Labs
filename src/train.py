import pandas as pd
import mlflow
from preprocessing import preprocess
from models import get_models, train_pipeline
from evaluation import evaluate_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn-mlops")

    logger.info("Loading dataset...")
    df = pd.read_csv("dataset/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    models = get_models()
    for model_name, model_instance in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = train_pipeline(model_name, model_instance, col_transf, X_train, y_train)
            evaluate_model(model_name, pipeline, X_test, y_test)

if __name__ == "__main__":
    main()
