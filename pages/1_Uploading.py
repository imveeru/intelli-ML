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
import pandas
if os.path.exists("sourcedata.csv"):
    print("Sourcedata")
    src=pd.read_csv("sourcedata.csv",index_col=None) 