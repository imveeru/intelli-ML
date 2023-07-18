import streamlit as st
st.set_page_config(
    page_title="IntelliML - Uploading",
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

import os
import pandas as pd
from fpdf import FPDF

if os.path.exists("sourcedata.csv"):
    print("Sourcedata")
    src=pd.read_csv("sourcedata.csv",index_col=None) 
    st.session_state["source_data"] = src

st.title("Upload your dataset for modelling!")
file=st.file_uploader("Upload your CSV file here.")
if file:
    df=pd.read_csv(file,index_col=None)
    df.to_csv("sourcedata.csv",index=None)
    
    st.session_state["pdf_report"]=FPDF(format='A4', unit='mm')
    st.session_state["pdf_report"].set_margin(15)
    st.session_state["pdf_report"].set_font("Helvetica",size=20,style="B")
    st.session_state["pdf_report"].set_auto_page_break(auto=True,margin=15)
    st.session_state["pdf_report"].add_page()
    effective_page_width = st.session_state["pdf_report"].w - 2*st.session_state["pdf_report"].l_margin

    st.session_state["pdf_report"].multi_cell(w=0,txt="# Page 1",ln=True)
    
    st.session_state["pdf_report"].set_font("Helvetica",size=12)
    st.session_state["pdf_report"].multi_cell(w=0,txt="# Page 1",ln=True)
    st.session_state["pdf_report"].ln(1.5)
    
    st.subheader("Dataset")
    st.dataframe(df)  
    st.caption("Now click on Analysis button to perform detailed data analysis on the uploaded dataset.")  
    st.session_state["pdf_report"].output("test.pdf")
    