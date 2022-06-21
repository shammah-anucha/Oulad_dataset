from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Student Prediction App",
    page_icon=im,
    layout="centered",
)

data = pd.read_csv(
    "https://raw.githubusercontent.com/shammah-anucha/Dataset/main/testing%20data.csv"
)
feature_list = data.columns

target_data = pd.read_csv(
    "https://raw.githubusercontent.com/shammah-anucha/Dataset/main/target%20data.csv"
)

st.header("Student Prediction App")
st.write(
    "This application is run by a Machine learning model that predicts the final results of students."
    "The final result classes are Distinction, Fail, Pass, and Withdrawn. For the purpose of testing, the testing data"
    "can be accessed by clicking the 'Download Test Data' button. A confusion matrix that shows the summary of the Artificial Neural Newtwork algorithm is displayed."
    "Three bar charts are plotted to show the predicted classes againsts some selected features. And finally a dataframe of the entire test data with the predicted classes is displayed"
)

try:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

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
        filtered_norm[cols_to_scale] = scaler.fit_transform(
            filtered_norm[cols_to_scale]
        )
except KeyError:
    st.warning(f"Headers should include {feature_list}")
except UnicodeDecodeError:
    st.warning(f"Please Upload a CSV File!")

try:

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

        st.subheader("Visualizations")

        # Confusion Matrix

        def categorise2(row):
            if row.final_result == "Distinction":
                return 0
            elif row.final_result == "Fail":
                return 1
            elif row.final_result == "Pass":
                return 2
            elif row.final_result == "Withdrawn":
                return 3

        target_data["final_result"] = target_data.apply(
            lambda row: categorise2(row), axis=1
        )

        def confusion_matrix_plot():
            cm = tf.math.confusion_matrix(
                labels=target_data.final_result, predictions=pred
            )

            plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True, fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Truth")

            return

        st.write(confusion_matrix_plot())

        # Count of Predicted classes

        def predicted_result_count():
            fig = px.bar(
                full_set,
                x=full_set.Prediction.value_counts().index,
                y=full_set.Prediction.value_counts(),
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
            )
            fig.show()

            return st.write(fig)

        def imd_results():
            imd = full_set[["imd_band", "Prediction"]]
            passed_10 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "0-10%")].index
            )
            withdrawn_10 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "0-10%")].index
            )
            fail_10 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "0-10%")].index
            )
            distinction_10 = len(
                imd[(imd.Prediction == "Distinction") & (imd.imd_band == "0-10%")].index
            )

            passed_20 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "10-20%")].index
            )
            withdrawn_20 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "10-20%")].index
            )
            fail_20 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "10-20%")].index
            )
            distinction_20 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "10-20%")
                ].index
            )

            passed_30 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "20-30%")].index
            )
            withdrawn_30 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "20-30%")].index
            )
            fail_30 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "20-30%")].index
            )
            distinction_30 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "20-30%")
                ].index
            )

            passed_40 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "30-40%")].index
            )
            withdrawn_40 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "30-40%")].index
            )
            fail_40 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "30-40%")].index
            )
            distinction_40 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "30-40%")
                ].index
            )

            passed_50 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "40-50%")].index
            )
            withdrawn_50 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "40-50%")].index
            )
            fail_50 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "40-50%")].index
            )
            distinction_50 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "40-50%")
                ].index
            )

            passed_60 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "50-60%")].index
            )
            withdrawn_60 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "50-60%")].index
            )
            fail_60 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "50-60%")].index
            )
            distinction_60 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "50-60%")
                ].index
            )

            passed_70 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "60-70%")].index
            )
            withdrawn_70 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "60-70%")].index
            )
            fail_70 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "60-70%")].index
            )
            distinction_70 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "60-70%")
                ].index
            )

            passed_80 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "70-80%")].index
            )
            withdrawn_80 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "70-80%")].index
            )
            fail_80 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "70-80%")].index
            )
            distinction_80 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "70-80%")
                ].index
            )

            passed_90 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "80-90%")].index
            )
            withdrawn_90 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "80-90%")].index
            )
            fail_90 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "80-90%")].index
            )
            distinction_90 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "80-90%")
                ].index
            )

            passed_100 = len(
                imd[(imd.Prediction == "Pass") & (imd.imd_band == "90-100%")].index
            )
            withdrawn_100 = len(
                imd[(imd.Prediction == "Withdrawn") & (imd.imd_band == "90-100%")].index
            )
            fail_100 = len(
                imd[(imd.Prediction == "Fail") & (imd.imd_band == "90-100%")].index
            )
            distinction_100 = len(
                imd[
                    (imd.Prediction == "Distinction") & (imd.imd_band == "90-100%")
                ].index
            )

            imd_result = {
                "Final_result": [
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                    "Pass",
                    "Withdrawn",
                    "Fail",
                    "Distinction",
                ],
                "Count of Final_result": [
                    passed_10,
                    withdrawn_10,
                    fail_10,
                    distinction_10,
                    passed_20,
                    withdrawn_20,
                    fail_20,
                    distinction_20,
                    passed_30,
                    withdrawn_30,
                    fail_30,
                    distinction_30,
                    passed_40,
                    withdrawn_40,
                    fail_40,
                    distinction_40,
                    passed_50,
                    withdrawn_50,
                    fail_50,
                    distinction_50,
                    passed_60,
                    withdrawn_60,
                    fail_60,
                    distinction_60,
                    passed_70,
                    withdrawn_70,
                    fail_70,
                    distinction_70,
                    passed_80,
                    withdrawn_80,
                    fail_80,
                    distinction_80,
                    passed_90,
                    withdrawn_90,
                    fail_90,
                    distinction_90,
                    passed_100,
                    withdrawn_100,
                    fail_100,
                    distinction_100,
                ],
                "imd_band": [
                    "0-10%",
                    "0-10%",
                    "0-10%",
                    "0-10%",
                    "10-20%",
                    "10-20%",
                    "10-20%",
                    "10-20%",
                    "20-30%",
                    "20-30%",
                    "20-30%",
                    "20-30%",
                    "30-40%",
                    "30-40%",
                    "30-40%",
                    "30-40%",
                    "40-50%",
                    "40-50%",
                    "40-50%",
                    "40-50%",
                    "50-60%",
                    "50-60%",
                    "50-60%",
                    "50-60%",
                    "60-70%",
                    "60-70%",
                    "60-70%",
                    "60-70%",
                    "70-80%",
                    "70-80%",
                    "70-80%",
                    "70-80%",
                    "80-90%",
                    "80-90%",
                    "80-90%",
                    "80-90%",
                    "90-100%",
                    "90-100%",
                    "90-100%",
                    "90-100%",
                ],
            }
            imd_result = pd.DataFrame(imd_result)

            fig = px.bar(
                imd_result,
                x="imd_band",
                y="Count of Final_result",
                color="Final_result",
                barmode="group",
            )

            return st.write(fig)

        placeholder = st.empty()

        with placeholder.container():
            fig_col1, fig_col2 = st.columns((2, 2))

            with fig_col1:
                st.markdown("###### Predicted Final Results")
                predicted_result_count()

            with fig_col2:
                st.markdown("###### Comparing Gender and final_result")
                gender_results()

        st.markdown("###### Comparing IMD Band and final_result")
        imd_results()

        st.markdown("### Detailed Data View")
        st.dataframe(full_set)

        st.write(f"Thank you! I hope you liked it.")
        st.write(
            f"If you want to see more advanced applications you can contact me on https://www.linkedin.com/in/shammahanucha/"
        )
except NameError:
    st.warning(f"Please Upload a CSV File!")
# except:
#     st.warning(f"Please Upload a CSV File!")
