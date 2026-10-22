"""Rebuild the bundled linear regression artifacts for the installed sklearn."""

from pathlib import Path
import pickle
import shutil

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    artifacts = Path(__file__).resolve().parent / "Artifacts"
    train = pd.read_csv(artifacts / "Train_Data.csv")
    test = pd.read_csv(artifacts / "Test_Data.csv")
    numerical = ["writing_score", "reading_score"]
    categorical = ["gender", "race_ethnicity", "parental_level_of_education",
                   "lunch", "test_preparation_course"]
    preprocessor = ColumnTransformer([
        ("num_pipeline", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat_pipelines", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False)),
        ]), categorical),
    ])
    X_train = preprocessor.fit_transform(train.drop(columns="math_score"))
    model = LinearRegression().fit(X_train, train["math_score"])
    predictions = model.predict(preprocessor.transform(test.drop(columns="math_score")))
    score = r2_score(test["math_score"], predictions)
    if score < 0.6:
        raise RuntimeError(f"Model quality too low to replace artifacts: R2={score:.4f}")
    for filename, obj in [("Preprocessor.pkl", preprocessor), ("Model.pkl", model)]:
        destination = artifacts / filename
        backup = artifacts / (filename + ".original")
        if destination.exists() and not backup.exists():
            shutil.copy2(destination, backup)
        with destination.open("wb") as output:
            pickle.dump(obj, output)
    print(f"Rebuilt model and preprocessor. Test R2: {score:.4f}")


if __name__ == "__main__":
    main()
