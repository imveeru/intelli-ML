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

def print_df(df,pdf,w):
    df = df.applymap(str)  # Convert all data inside dataframe into string type

    columns = [list(df)]  # Get list of dataframe columns
    rows = df.values.tolist()  # Get list of dataframe rows
    data = columns + rows  # Combine columns and rows in one list
    pdf.set_font("Arial",size=7)
    with pdf.table(line_height=pdf.font_size * 2.5,
                text_align="CENTER",
                width=w) as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

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
        st.subheader("Experiment Settings")
        st.dataframe(setup_df)
        
        setup_response=ask_llm(setup_prompt)
        
        
        st.write(setup_response)
        
        st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
        st.session_state["pdf_report"].multi_cell(w=0,txt="Machine Learning experiment settings",ln=True,align="L")
        st.session_state["pdf_report"].ln(2.5)
        st.session_state["pdf_report"].set_font("Arial",size=9)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=setup_response,ln=True,align="J")
        st.session_state["pdf_report"].ln(5)
        
        print_df(setup_df,st.session_state["pdf_report"],st.session_state["print_w"])
        st.session_state["pdf_report"].ln(5)
        
        
    with st.spinner("Fitting the data in various models..."):
        best_model=compare_models()
        compare_df=pull()
        
        model_exp_prompt=f'''
        {compare_df["Model"].values}
        
        The above given is an array of Machine Learning models. Write a single line explanation for each model.
        Do not exceed 150 words.
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
        
        st.markdown("#### Machine learning models used")
        st.write(model_exp_response)
        st.markdown("#### Machine learning model performance comparison")
        st.dataframe(compare_df)
        st.write(model_comparison_response)
        #st.pyplot(plot_model(best_model, plot = 'residuals'))
        save_model(best_model,"best_model")
        
        
        st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
        st.session_state["pdf_report"].multi_cell(w=0,txt="Machine learning models used",ln=True,align="L")
        st.session_state["pdf_report"].ln(2.5)
        st.session_state["pdf_report"].set_font("Arial",size=9)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=model_exp_response,ln=True,align="J")
        st.session_state["pdf_report"].ln(5)
        
        st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
        st.session_state["pdf_report"].multi_cell(w=0,txt="Machine learning model performance comparison",ln=True,align="L")
        st.session_state["pdf_report"].ln(2.5)
        
        print_df(compare_df,st.session_state["pdf_report"],st.session_state["print_w"])
        st.session_state["pdf_report"].ln(5)
        
        st.session_state["pdf_report"].set_font("Arial",size=9)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=model_comparison_response,ln=True,align="J")
        st.session_state["pdf_report"].ln(5)
       