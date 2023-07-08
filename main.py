import streamlit as st
import pandas as pd
import os
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

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
    choice=st.radio("Navigation",["Upload","Analysis","Modelling","Download"])
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

if choice=="Analysis":
    st.title("Automated Exploratory Data Analysis")
    profile_report=df.profile_report()
    st_profile_report(profile_report)

if choice=="Modelling":
    pass

if choice=="Download":
    pass