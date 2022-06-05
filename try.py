# # import streamlit as st
import pandas as pd

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
import numpy as np

# # import tensorflow as tf
# # from tensorflow import keras

data = pd.read_csv("/home/shammah/Downloads/full_set.csv")
data = data.drop(columns=["sum_click"])


def new_inputs():

    region = input("Please enter region: ")
    highest_education = input("Please enter highest_education: ")
    imd_band = input("Please enter imd_band: ")
    age_band = input("Please enter age_band: ")
    disability = input("Please enter disability: ")
    gender = input("Please enter gender: ")
    score = int(input("Please enter score: "))
    number_of_previous_attempts = int(
        input("Please enter number_of_previous_attempts: ")
    )
    studied_credits = int(input("Please enter studied_credits: "))

    data1 = {
        "score": score,
        "number_of_previous_attempts": number_of_previous_attempts,
        "studied_credits": studied_credits,
    }

    data1 = pd.DataFrame(data1, index=[0])

    data2 = pd.get_dummies(
        region,
        highest_education,
        imd_band,
        age_band,
        disability,
        gender,
    )
    print(data2)
    new_data = pd.concat([data1, data2])
    print(new_data)

    return new_data


# if __name__ == "__main__":
#     print(new_inputs())


# 'region_East Anglian Region'	region_East Midlands Region	region_Ireland	region_London Region	region_North Region	region_North Western Region	region_Scotland	region_South East Region	region_South Region	region_South West Region	region_Wales	region_West Midlands Region	region_Yorkshire Region


data4 = pd.read_csv("/home/shammah/Downloads/student_data.csv")
feature_list = data4.columns
d = pd.DataFrame(0, index=[0], columns=feature_list)
# print(d)


region = np.unique(data["region"])
input_region = input("Please enter region: ")

disability = np.unique(data["disability"])
input_disability = input("Is student disabled?: ")

highest_education = np.unique(data["highest_education"])
input_highest_education = input("Please enter highest_education: ")

imd_band = np.unique(data["imd_band"])
input_imd_band = input("Please enter imd_band: ")

age_band = np.unique(data["age_band"])
input_age_band = input("Please enter age_band: ")

gender = np.unique(data["gender"])
input_gender = input("Please enter gender: ")

input_score = int(input("Please enter score: "))
input_number_of_previous_attempts = int(
    input("Please enter number_of_previous_attempts: ")
)
input_studied_credits = int(input("Please enter studied_credits: "))


for i in disability:
    if i == input_disability:
        d.at[0, "disability" + "_" + input_disability] = 1

for i in gender:
    if i == input_gender:
        d.at[0, "gender" + "_" + input_gender] = 1

for i in imd_band:
    if i == input_imd_band:
        d.at[0, "imd_band" + "_" + input_imd_band] = 1

for i in age_band:
    if i == input_age_band:
        d.at[0, "age_band" + "_" + input_age_band] = 1

for i in highest_education:
    if i == input_highest_education:
        d.at[0, "highest_education" + "_" + input_highest_education] = 1

for i in region:
    if i == input_region:
        d.at[0, "region" + "_" + input_region] = 1

d.at[0, "num_of_prev_attempts"] = input_number_of_previous_attempts
d.at[0, "score"] = input_score
d.at[0, "studied_credits"] = input_studied_credits


print(d)
