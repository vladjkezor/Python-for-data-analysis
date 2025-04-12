import os

import joblib
import numpy as np
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
    X, y = train.drop(TARGET_COL, axis=1), train[TARGET_COL[0]]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS)
        ])

    model = RandomForestRegressor

    pipeline = Pipeline(steps=[
        ('processing', preprocessor),
        ('model', model(random_state=42))
    ])



    pipeline.fit(X, y)

    joblib.dump(pipeline, "model/my_model.pkl")


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    return model

