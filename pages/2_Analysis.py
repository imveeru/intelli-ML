import streamlit as st
st.set_page_config(
    page_title="IntelliML - Analysis",
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

import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.title("Automated Exploratory Data Analysis")
src=st.session_state["source_data"]
profile = ProfileReport(src,explorative=True)
st_profile_report(profile) 