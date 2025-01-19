 # Application Code

import numpy as np
import pickle
import streamlit as st

# Load the model
with open("LR_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

def survival(input_data):
    try:
        inp_data_array = np.array(input_data, dtype=float)
        inp_data_reshaped = inp_data_array.reshape(1, -1)
        pred_in = loaded_model.predict(inp_data_reshaped)
        if pred_in[0] == 0:
            return "The Person did not SURVIVE"
        else:
            return "The Person Survived"
    except ValueError:
        return "Invalid input. Please enter numeric values for all fields."

def main():
    st.title("Titanic Survival Web Application")
    st.header("Enter the details of a passenger")
    
    # Input values from the user
    Pclass = st.text_input("Enter Pclass (1, 2, 3)", value="1")
    Sex = st.text_input("Enter Sex of a person (0:male or 1:female)", value="1")
    Age = st.text_input("Enter Age of the person (0-80)", value="25")
    SibSp = st.text_input("Enter SibSp (0-5)", value="0")
    Parch = st.text_input("Enter Parch (0-6)", value="0")
    Fare = st.text_input("Enter Fare (0-512)", value="50")
    c = st.text_input("Enter c from which place (0, 1)", value="0")
    q = st.text_input("Enter q from which place (0, 1)", value="1")
    s = st.text_input("Enter s from which place (0, 1)", value="0")

    answer = ""  # Placeholder for prediction result

    if st.button("Predict Survival"):
        answer = survival([Pclass, Sex, Age, SibSp, Parch, Fare,c,q,s])
        st.success(answer)

if __name__ == '__main__':
    main()
