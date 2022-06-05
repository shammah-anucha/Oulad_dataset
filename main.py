import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.header("Student Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv(
    "https://raw.githubusercontent.com/shammah-anucha/Dataset/main/full_set.csv"
)
data = data.drop(columns=["sum_click"])

data4 = pd.read_csv(
    "https://raw.githubusercontent.com/shammah-anucha/Dataset/main/student_data.csv"
)
feature_list = data4.columns
d = pd.DataFrame(0, index=[0], columns=feature_list)


# load model
ann_model = tf.keras.models.load_model("best_model.h5")


if st.checkbox("Show Training Dataframe"):
    data

c_region = np.unique(data["region"])
c_disability = np.unique(data["disability"])
c_highest_education = np.unique(data["highest_education"])
c_imd_band = np.unique(data["imd_band"])
c_age_band = np.unique(data["age_band"])
c_gender = np.unique(data["gender"])


st.subheader("Please select relevant Student data!")
left_column, right_column = st.columns(2)
with left_column:
    gender = st.radio("Student Gender:", c_gender)
with left_column:
    region = st.radio("Student Region:", c_region)
with left_column:
    highest_education = st.radio("Student Highest Education:", c_highest_education)
with left_column:
    imd_band = st.radio("Student Imd band:", c_imd_band)
with left_column:
    age_band = st.radio("Student Age band:", c_age_band)
with left_column:
    disability = st.radio("Is Student Disabled?:", c_disability)


number_of_previous_attempts = st.slider(
    "Number of previous attempts", 0, max(data["num_of_prev_attempts"]), 1
)
studied_credits = st.slider("Studied credits", 0, max(data["studied_credits"]), 1)
score = st.slider("Score", 0, max(data["score"]), 1)


if st.button("Make Prediction"):
    for i in disability:
        if i == c_disability:
            d.at[0, "disability" + "_" + c_disability] = 1
    for i in gender:
        if i == c_gender:
            d.at[0, "gender" + "_" + c_gender] = 1
    for i in imd_band:
        if i == c_imd_band:
            d.at[0, "imd_band" + "_" + c_imd_band] = 1
    for i in age_band:
        if i == c_age_band:
            d.at[0, "age_band" + "_" + c_age_band] = 1
    for i in highest_education:
        if i == c_highest_education:
            d.at[0, "highest_education" + "_" + c_highest_education] = 1
    for i in region:
        if i == c_region:
            d.at[0, "region" + "_" + c_region] = 1

    d.at[0, "num_of_prev_attempts"] = number_of_previous_attempts
    d.at[0, "score"] = score
    d.at[0, "studied_credits"] = studied_credits

    inputs = d

    prediction = ann_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your student's result is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    st.write(
        f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)"
    )
