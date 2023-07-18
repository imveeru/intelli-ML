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
target=st.session_state["target"]

if st.button("Train the model"):
    with st.spinner("Setting the experiment parameters..."):
        setup(src,target=target)
        setup_df=pull()
        #st.text(setup_df.columns)
        setup_df=setup_df[setup_df['Description'].ne('Session id')]
        setup_df=setup_df[setup_df['Description'].ne('USI')]
        setup_df=setup_df[setup_df['Description'].ne('Experiment Name')]
        setup_df=setup_df[setup_df['Description'].ne('Log Experiment')]
        #setup_df.drop(['Session id'],inplace=True)
        st.subheader("Experiment Settings")
        st.dataframe(setup_df)
    with st.spinner("Fitting the data in various models..."):
        best_model=compare_models()
        compare_df=pull()
        st.info("Model Comparison")
        st.dataframe(compare_df)
        save_model(best_model,"best_model")