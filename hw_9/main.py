import os

import pandas as pd
import streamlit as st

from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Price predictor",
)
NUM_COLS = ['total_square', 'rooms', 'floor']
model_path = 'rf_fitted.pkl'

square = st.sidebar.number_input("Total square of flat", 8, 2200, 42, 10)

rooms = st.sidebar.slider(
    "How many rooms",
    0, 20, 1,
)
floor = st.sidebar.slider('On witch floor is it?', 1, 100, 1)

# create input DataFrame
inputDF = pd.DataFrame(
    {
        "total_square": square,
        "rooms": rooms,
        "floor": floor,
    },
    index=[0],
)

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model('rf_fitted.pkl')

st.image("img/pic.jpg", use_container_width=True)

if st.button("Predict Price"):
    pred = model.predict(inputDF)[0]
    st.success(f"Predicted Price: {round(pred)} $")
