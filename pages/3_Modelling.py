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
import google.generativeai as palm
from dotenv import dotenv_values

config = dotenv_values(".env") 

palm.configure(api_key=config["GOOGLE_PALM_API_KEY"])

def ask_llm(prompt):
    defaults = {
        'model': 'models/text-bison-001',
        'temperature': 0.7,
        'candidate_count': 1,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 1024,
        'stop_sequences': [],
        'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    }
    
    res=palm.generate_text(**defaults, prompt=prompt)
    
    return res.result

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
        
        setup_prompt=f'''
        {setup_df}
                
        The above given are the settings of an Machine learning experiment.
        Write a paragraph using the given data. Make sure the paragraph is in an academic tone. 
        Strictly do not exceed 200 words.
        '''
        
        setup_response=ask_llm(setup_prompt)
        
        st.subheader("Experiment Settings")
        st.dataframe(setup_df)
        st.write(setup_response)
        
    with st.spinner("Fitting the data in various models..."):
        best_model=compare_models()
        compare_df=pull()
        
        model_exp_prompt=f'''
        {compare_df["Model"].values}
        
        The above given is an array of Machine Learning models. Write a single line explanation for each model.
        '''
        
        model_comparison_prompt=f'''
        {compare_df}

        The above given are the list of Machine Learning algorithms and metrics like Mean Absolute Error(MAE), Mean Square Error(MSE), Root Mean Square Error(RMSE), R2 Score, RMSLE and MAPE in which a dataset is trained and tested.

        Write a paragraph by comparing the performance of all the models by using the given metrics.

        Through comparing {best_model} is the best performing model.

        Write the possible reason why it'd have performed well.
        Do not forget to include the comparison of other models.
        Make sure the content is in academic tone.
        '''
        
        model_exp_response=ask_llm(model_exp_prompt)
        model_comparison_response=ask_llm(model_comparison_prompt)
        
        st.info("Model Comparison")
        st.write(model_exp_response)
        st.dataframe(compare_df)
        st.write(model_comparison_response)
        save_model(best_model,"best_model")