import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns


def main():
    numbers_df = load_data()

    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    if page == "Homepage":
        st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(numbers_df)
    elif page == "Exploration":
        st.title("Data Exploration")

        number = st.slider("Choose picture to show: ", min_value=1,
                               max_value=100, value=35, step=1)

        numbers_x_df = numbers_df.drop(labels=['label'], axis=1)
        numbers_x_df = numbers_x_df.values.reshape(-1, 28, 28, 1)
        numbers_x_df = numbers_x_df / 255.0
        plt.imshow(numbers_x_df[number][:, :, 0])

        st.pyplot()

        numbers_y_df = numbers_df['label']
        sns.countplot(numbers_y_df)
        plt.suptitle("Распределение картинок по номерам")
        st.pyplot()


@st.cache
def load_data():
    numbers_df = pd.read_csv('train.csv')
    return numbers_df


if __name__ == "__main__":
    main()
