import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUM_COLS = ['total_square', 'rooms', 'floor']
TARGET_COL = ['price']


def prepare_data():
    train = pd.read_csv("data/realty_data.csv")

    train["rooms"] = train["rooms"].fillna(0)
    train.dropna(subset='city', inplace=True)

    train = train[NUM_COLS + TARGET_COL]

    return train


def train_model(train):
    X, y = train.drop(TARGET_COL, axis=1), train[TARGET_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS)
        ])

    model = RandomForestRegressor

    pipeline = Pipeline(steps=[
        ('processing', preprocessor),
        ('model', model(random_state=42))
    ])

    param_dist = {
        "model__n_estimators": np.arange(200, 2001, 200),
        "model__max_depth": list(np.arange(10, 101, 10)) + [None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__bootstrap": [True, False]
    }

    rf = RandomizedSearchCV(
        pipeline,
        param_dist,
        cv=5,
        n_iter=50,
        n_jobs=4,
        verbose=1,
    )

    rf.fit(X, y)

    with open('rf_fitted.pkl', 'wb') as file:
        pickle.dump(rf, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model
