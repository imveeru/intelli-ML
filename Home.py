import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import *

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
    print("Sourcedata")
    src=pd.read_csv("sourcedata.csv",index_col=None)  

if choice == "Upload":
    st.title("Upload your dataset for modelling!")
    file=st.file_uploader("Upload your CSV file here.")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
    
        if st.button("Next >",key="p1"):
            choice="Analysis"
 
if choice=="Analysis":
    st.title("Automated Exploratory Data Analysis")
    profile = ProfileReport(src,explorative=True)
    st_profile_report(profile)
    
    if st.button("Next >",key="p21"):
        choice="Modelling"
    
    if st.button("< Previous",key="p22"):
        choice="Upload"

if choice=="Modelling":
    st.title("Modelling")
    target=st.selectbox("Select the target variable",src.columns)
    if st.button("Train the model"):
        setup(src,target=target)
        setup_df=pull()
        st.info("Experiment Settings")
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.info("Model Comparison")
        st.dataframe(compare_df)
        save_model(best_model,"best_model")
        #plot_model(best_model,plot='feature')
        
    if st.button("Next >",key="p31"):
        choice="Download"

    if st.button("< Previous",key="p32  "):
        choice="Analysis"

if choice=="Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the best model",f,"best_model.pkl")