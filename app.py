#### LOCAL APP
#### Sentiment analysis application

#### Import packages
import streamlit as st
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

#### Add header to describe app
st.title("LinkedIn User Analysis Prediction")
st.image("https://leadgenera.lg-cms.com/wp-content/uploads/2022/09/Is-LinkedIn-the-Ultimate-Business-Tool-Lead-Genera.png")
st.write("image source: Google")

st.subheader("Please provide your responses to the following:")
 ## Age
age = st.slider('Please drag the slider to select your age:', 0, 97)
st.write("Age:", age)

## Gender
gender = st.radio('Gender:', ['Female', 'Male', 'Other'])
st.write(f"You selected: {gender}")

if gender == "Female":
    gender = 1
else:
    gender = 0

#st.write(f"You selected: (post-conversion): {gender}")

## Married
married = st.radio('Are you married?', ['Yes', 'No'])
st.write(f"You selected: {married}")

if married == "Yes":
    married = 1
else:
    married = 0

#st.write(f"Marital (post-conversion): {married}")

## Parent 
parent = st.radio('Are you a parent?', ['Yes', 'No'])
st.write(f"You selected: {parent}")

if parent == "Yes":
    parent = 1
else:
    parent = 0

#st.write(f"Parent (post-conversion): {parent}")

## Income
income = st.radio('Please select your income range:',
['Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000', '40 to under $50,000', 
'50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'])
st.write(f"You selected:{income}")

if income =="Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
else:
    income = 9

#st.write(f"Income (post-conversion): {income}")

## Education
education = st.radio('Please select your highest level of education:', ['Less than high school', 'High school incomplete', 'High school graduate', 
'Some college, no degree', 'Two-year associate degree', 'Four-year college or university degree/Bachelor’s degree', 
'Some postgraduate or professional schooling', 'Postgraduate or professional degree'])
st.write(f"You selected: {education}")

if education == "Less than high school":
    education = 1
elif education == "High school incomplete":
    education = 2
elif education == "High school graduate":
    education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Two-year associate degree":
    education = 5
elif education == "Four-year college or university degree/Bachelor’s degree":
    education = 6
elif education == "Some postgraduate or professional schooling":
    education = 7
else:
    education = 8

#st.write(f"Education (post-conversion): {education}")

import pandas as pd
s = pd.read_csv("social_media_usage.csv")

ss = pd.DataFrame({
        "parent":np.where(s["par"] >= 8, np.nan,
                          np.where(s["par"] == 1, 1, 0)),
        "married":np.where(s["marital"] >= 8, np.nan,
                          np.where(s["marital"] == 1, 1, 0)),
        "female":np.where(s["gender"] >= 98, np.nan,
                          np.where(s["gender"] == 2, 1, 0)),
        "age":np.where(s["age"] >= 98, np.nan,
                   s["age"]),
        "income":np.where(s["income"] >= 98, np.nan,
                      s["income"]),
        "education":np.where(s["educ2"] >= 98, np.nan,
                         s["educ2"]),
        "sm_li":np.where(s["sample"] >= 8, np.nan,
                          np.where(s["sample"] == 1, 1, 0)),})
ss = ss.dropna()
#st.title("Clean Data")
#st.write(ss)

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987)

# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')
# Fit algorithm to training data
lr.fit(X_train, y_train)

# Evaluate Model Performance
# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# Compare those predictions to the actual test data using a confusion matrix (positive class=1)

#confusion_matrix(y_test, y_pred)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

# New data for features: parent, married, female, age, income, education
user = [age, gender, married, parent, income, education]

# Predict class, given input features
predicted_class = lr.predict([user])

if predicted_class == 1:
    predicted_class_label = "a LinkedIn User"
else:
    predicted_class_label = "Not a LinkedIn User"

# Generate probability of positive class (=1)
probs = lr.predict_proba([user])

if st.button('Predict'):
    st.write(f"Probability that you are a LinkedIn User: {probs[0][1].round(2)}")
    st.write(f"This person is: {predicted_class_label}")
else:
    st.write(" ")

st.sidebar.subheader("Thank you for using the LinkedIn User Prediction App! This app takes user input and predicts the probability that the user is a LinkedIn user. Please enter your information to see the prediction!")
st.sidebar.write("Created by: Sydney Peirce for OPIM-607")


