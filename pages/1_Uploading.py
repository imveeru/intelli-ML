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

def print_df(df,pdf):
    df = df.applymap(str)  # Convert all data inside dataframe into string type

    columns = [list(df)]  # Get list of dataframe columns
    rows = df.values.tolist()  # Get list of dataframe rows
    data = columns + rows  # Combine columns and rows in one list
    pdf.set_font("Helvetica",size=7)
    with pdf.table(cell_fill_color=200,  # grey
                cell_fill_mode="ROWS",
                line_height=pdf.font_size * 2.5,
                text_align="CENTER",
                width=160) as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

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
    st.session_state["pdf_report"].set_auto_page_break(auto=True,margin=15)
    st.session_state["pdf_report"].add_page()
    effective_page_width = st.session_state["pdf_report"].w - 2*st.session_state["pdf_report"].l_margin

    st.session_state["pdf_report"].set_font("Helvetica",size=24,style="B")
    st.session_state["pdf_report"].multi_cell(w=0,txt="IntelliML Report",ln=True,align="C")
    st.session_state["pdf_report"].ln(10)
    
    st.session_state["pdf_report"].set_font("Helvetica",size=12)
    st.session_state["pdf_report"].multi_cell(w=effective_page_width,txt="Sample Datset",ln=True,align="J")
    st.session_state["pdf_report"].ln(2.5)
    print_df(df.head(),st.session_state["pdf_report"])
    
    st.subheader("Dataset")
    st.dataframe(df.head())  
    st.caption("Now click on Analysis button to perform detailed data analysis on the uploaded dataset.")  
    st.session_state["pdf_report"].output("test.pdf")
    