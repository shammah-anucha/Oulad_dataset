import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sn

st.title("Hello world!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Add some matplotlib code !

    fig = plt.figure(figsize=(10, 8))
    sn.barplot(
        x=df.final_result.value_counts().index,
        y=df.final_result.value_counts(),
        data=df,
        color="grey",
    )
    plt.title("Final Result Count", fontsize=20)
    plt.show()

    st.write(fig)
