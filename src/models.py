from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlflow.models import infer_signature
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)

def get_models():
    """Return a dictionary of models to train."""
    return {
        "logreg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    }


def train_pipeline(model_name, model_instance, col_transf, X_train, y_train):
    """Create a pipeline, train the model, return trained pipeline."""
    logger.info(f"Training {model_name}...")
    pipeline = Pipeline([
        ("model", model_instance)
    ])
    pipeline.fit(X_train, y_train)

    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.log_model(
        pipeline,
        "pipeline",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    logger.info(f"{model_name} training completed.")
    return pipeline
