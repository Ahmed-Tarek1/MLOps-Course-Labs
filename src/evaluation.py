from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model_name, pipeline, X_test, y_test):
    """Evaluate the model, log metrics and confusion matrix."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    mlflow.log_metrics(metrics)
    mlflow.set_tag("model_type", model_name)

    logger.info(f"Results for {model_name}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    conf_mat = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    
    ax.set_title(f"Confusion Matrix: {model_name}")
    
    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close(fig)
