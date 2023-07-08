import streamlit as st
import pandas as pd
import os

#page config
st.set_page_config(
    page_title="IntelliML",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

#hide streamlit default
hide_st_style ='''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    st.title("🤖 IntelliML")
    choice=st.radio("Navigation",["Upload","Profiling","ML Modelling","Download the model"])
    st.info("This application helps you to build an automated ML pipeline.")

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv",index_col=None)  

if choice == "Upload":
    st.title("Upload your dataset for modelling!")
    file=st.file_uploader("Upload your CSV file here.")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("soucedata.csv",index=None)
        st.dataframe(df)

if choice=="Profiling":
    pass

if choice=="ML Modelling":
    pass

if choice=="Download the model":
    pass