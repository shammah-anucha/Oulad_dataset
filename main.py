import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


# st.text_input("Enter your Name: ", key="name")

st.set_page_config(
    page_title="Student Prediction App",
    page_icon="✅",
    layout="wide",
)

st.header("Student Prediction App")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # load model
    ann_model = tf.keras.models.load_model("new_ann_model.h5")

    # Data Encoding
    sws_enc = pd.get_dummies(
        df,
        columns=[
            "code_module",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "disability",
            "gender",
            "code_presentation",
        ],
    )
    filtered = sws_enc.drop(columns=["id_student", "disability_N", "gender_F"])

    # Normalization
    scaler = MinMaxScaler()
    filtered_norm = filtered.copy()
    cols_to_scale = ["num_of_prev_attempts", "studied_credits", "score_100"]
    filtered_norm[cols_to_scale] = scaler.fit_transform(filtered_norm[cols_to_scale])


if st.button("Make Prediction"):

    inputs = filtered_norm

    prediction = ann_model.predict(inputs)

    pred = np.argmax(prediction, axis=1)
    pred_string = np.array(pred)

    result = pd.DataFrame(pred_string, columns=["Prediction"])

    frames = [df, result]
    full_set = pd.concat(frames, axis=1)

    def categorise(row):
        if row.Prediction == 0:
            return "Distinction"
        elif row.Prediction == 1:
            return "Fail"
        elif row.Prediction == 2:
            return "Pass"
        elif row.Prediction == 3:
            return "Withdrawn"

    full_set["Prediction"] = full_set.apply(lambda row: categorise(row), axis=1)
    st.write(full_set)

    st.subheader("Visualizations")

    # Count of Predicted classes

    def predicted_result_count():
        fig = px.bar(
            full_set,
            x=full_set.Prediction.value_counts().index,
            y=full_set.Prediction.value_counts(),
            title="Predicted Final Results",
        )
        return st.write(fig)

    # comparing gender and Predicted final_result

    def gender_results():
        gen = full_set[["Prediction", "gender"]]
        passed_m = len(gen[(gen.Prediction == "Pass") & (gen.gender == "M")].index)
        passed_f = len(gen[(gen.Prediction == "Pass") & (gen.gender == "F")].index)
        withdrawn_m = len(
            gen[(gen.Prediction == "Withdrawn") & (gen.gender == "M")].index
        )
        withdrawn_f = len(
            gen[(gen.Prediction == "Withdrawn") & (gen.gender == "F")].index
        )
        fail_m = len(gen[(gen.Prediction == "Fail") & (gen.gender == "M")].index)
        fail_f = len(gen[(gen.Prediction == "Fail") & (gen.gender == "F")].index)
        distinction_m = len(
            gen[(gen.Prediction == "Distinction") & (gen.gender == "M")].index
        )
        distinction_f = len(
            gen[(gen.Prediction == "Distinction") & (gen.gender == "F")].index
        )
        gender_result = {
            "Predicted Result": [
                "Pass",
                "Pass",
                "Withdrawn",
                "Withdrawn",
                "Fail",
                "Fail",
                "Distinction",
                "Distinction",
            ],
            "Count of Final_result": [
                passed_m,
                passed_f,
                withdrawn_m,
                withdrawn_f,
                fail_m,
                fail_f,
                distinction_m,
                distinction_f,
            ],
            "Gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
        gender_result = pd.DataFrame(gender_result)

        fig = px.bar(
            gender_result,
            x="Predicted Result",
            y="Count of Final_result",
            color="Gender",
            barmode="group",
            height=400,
        )
        fig.show()
        fig.update_layout(title_text="Comparing gender and final_result")

        return st.write(fig)

    placeholder = st.empty()

    with placeholder.container():
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            predicted_result_count()

        with fig_col2:
            gender_results()

    st.write(f"Thank you! I hope you liked it.")
    st.write(
        f"If you want to see more advanced applications you can contact me on https://www.linkedin.com/in/shammahanucha/"
    )

# {st.session_state.name}
