#sample code for streamlitui
import streamlit as st
import pandas as pd
st.title("Streamlit first code")
st.header("First form in streamlit")
x= st.text_input("Enter your name:")
y = st.number_input("Enter age:")
z = st.radio("Choose you degree:",['cse','eee','ece','mech','civil'])
a = st.slider("rank",1,10000)
b = st.date_input("Date joined:")
st.write("name: " + x)
st.write("age: " + str(y))
st.write("degree: " + str(z))
st.write("rank: " + str(a))
st.write("date joined: " + str(b))
