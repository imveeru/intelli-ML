import streamlit as st
import pandas as pd

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
    

if choice == "Upload":
    pass

if choice=="Profiling":
    pass

if choice=="ML Modelling":
    pass

if choice=="Download the model":
    pass