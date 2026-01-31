import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebalance(data: pd.DataFrame) -> pd.DataFrame:
    """Balance target classes by downsampling the majority class."""
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj, churn_min = churn_0, churn_1
    else:
        churn_maj, churn_min = churn_1, churn_0

    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df: pd.DataFrame):
    """Preprocess features, split train/test, return ColumnTransformer and datasets."""
    logger.info("Starting preprocessing...")
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    data = df.loc[:, filter_feat]
    data_bal = rebalance(data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough"
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    mlflow.sklearn.log_model(col_transf, artifact_path="preprocessor")
    logger.info("Preprocessing done.")
    return col_transf, X_train, X_test, y_train, y_test
