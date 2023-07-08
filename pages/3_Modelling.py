import streamlit as st
st.set_page_config(
    page_title="IntelliML - Modelling",
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

from pycaret.regression import *

st.title("Modelling")
src=st.session_state["source_data"]
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