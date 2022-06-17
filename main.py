import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

""" The visualizations should include the demographics of the students"""

st.header("Student Prediction App")
st.text_input("Enter your Name: ", key="name")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)


# data4 = pd.read_csv(
#     "https://raw.githubusercontent.com/shammah-anucha/Oulad_dataset/main/testing_header.csv?token=GHSAT0AAAAAABTRCNKPDGX5ZSVPNYSDB5AUYVMET5Q"
# )
# feature_list = data4.columns
# d = pd.DataFrame(0, index=[0], columns=feature_list)


# load model
ann_model = tf.keras.models.load_model("new_ann_model.h5")


# if st.checkbox("Show Training Dataframe"):
#     data

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


# c_region = np.unique(data["region"])
# c_disability = np.unique(data["disability"])
# c_highest_education = np.unique(data["highest_education"])
# c_imd_band = np.unique(data["imd_band"])
# c_age_band = np.unique(data["age_band"])
# c_gender = np.unique(data["gender"])


# st.subheader("Please select relevant Student data!")
# left_column, right_column = st.columns(2)
# with left_column:
#     gender = st.radio("Student Gender:", c_gender)
# with left_column:
#     region = st.radio("Student Region:", c_region)
# with left_column:
#     highest_education = st.radio("Student Highest Education:", c_highest_education)
# with left_column:
#     imd_band = st.radio("Student Imd band:", c_imd_band)
# with left_column:
#     age_band = st.radio("Student Age band:", c_age_band)
# with left_column:
#     disability = st.radio("Is Student Disabled?:", c_disability)


# number_of_previous_attempts = st.slider(
#     "Number of previous attempts", 0, max(data["num_of_prev_attempts"]), 1
# )
# studied_credits = st.slider("Studied credits", 0, max(data["studied_credits"]), 1)
# score = st.slider("Score", 0.0, max(data["score"]), 1.0)


if st.button("Make Prediction"):
    #     for i in disability:
    #         if i == disability:
    #             d.at[0, "disability" + "_" + c_disability] = 1
    #     for i in gender:
    #         if i == gender:
    #             d.at[0, "gender" + "_" + c_gender] = 1
    #     for i in imd_band:
    #         if i == imd_band:
    #             d.at[0, "imd_band" + "_" + c_imd_band] = 1
    #     for i in age_band:
    #         if i == age_band:
    #             d.at[0, "age_band" + "_" + c_age_band] = 1
    #     for i in highest_education:
    #         if i == highest_education:
    #             d.at[0, "highest_education" + "_" + c_highest_education] = 1
    #     for i in region:
    #         if i == region:
    #             d.at[0, "region" + "_" + c_region] = 1

    #     d.at[0, "num_of_prev_attempts"] = number_of_previous_attempts
    #     d.at[0, "score"] = score
    #     d.at[0, "studied_credits"] = studied_credits

    #     d = d.drop(
    #         columns=[
    #             "final_result_Distinction",
    #             "final_result_Fail",
    #             "final_result_Pass",
    #             "sum_click",
    #         ]
    #     )

    inputs = filtered_norm

    prediction = ann_model.predict(inputs)

    pred = np.argmax(prediction, axis=1)
    pred_string = np.array(pred)
    # st.write(pred_string)

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
    fig = plt.figure(figsize=(10, 8), facecolor="#262730")
    ax = plt.axes()
    sn.barplot(
        x=full_set.Prediction.value_counts().index,
        y=full_set.Prediction.value_counts(),
        data=full_set,
        color="#F63366",
    )
    ax.set_title("Predicted Final Results", fontsize=20, color="white")
    ax.set_facecolor("#262730")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines[["left", "top", "right", "bottom"]].set_color("white")
    plt.show()

    st.write(fig)
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

        # comparing gender and final_result
        # Set the figure size
        figs = plt.figure(figsize=(8, 8), facecolor="#262730")
        # grouped barplot
        sn.barplot(
            x="Predicted Result",
            y="Count of Final_result",
            hue="Gender",
            data=gender_result,
            color="#F63366",
        )
        # ax1 = plt.axes()
        # ax1.set_title("Comparing gender and final_result", fontsize=20, color="white")
        ax.set_facecolor("#262730")
        # ax1.tick_params(axis="x", colors="white")
        # ax1.tick_params(axis="y", colors="white")
        # ax1.spines[["left", "top", "right", "bottom"]].set_color("white")
        plt.show()

        return st.write(figs)

    gender_results()

    # st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    # st.write(
    #     f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)"
    # )
