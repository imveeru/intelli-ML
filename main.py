import streamlit as st
import pandas as pd

st.title("Hello World")

#hide streamlit default
hide_st_style ='''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

#page config
st.set_page_config(
    page_title="CulinaryCrafter",
    page_icon="🥣",
    initial_sidebar_state="expanded",
)