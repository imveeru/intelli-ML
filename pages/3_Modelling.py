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
import json
import google.generativeai as palm
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel
from dotenv import dotenv_values

config = dotenv_values(".env") 

service_account_info=json.loads(config["GOOGLE_APPLICATION_CREDENTIALS"])

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

# Initialize Google AI Platform with project details and credentials
aiplatform.init(
    credentials=my_credentials,
)

# with open("service_account.json", encoding="utf-8") as f:
#     project_json = json.load(f)
project_id = service_account_info["project_id"]


# Initialize Vertex AI with project and location
vertexai.init(project=project_id, location="us-central1")

def ask_llm(prompt):
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        prompt,
        **parameters
    )
    
    return response.text

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

if st.button("Train and Test various Machine Learning models"):
    with st.spinner("Setting up the experiment parameters..."):
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
        Strictly do not exceed 150 words.
        '''
        st.subheader("Experiment Settings")
        st.dataframe(setup_df)
        
        setup_response=ask_llm(setup_prompt)
        
        
        st.write(setup_response)
        
        st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
        st.session_state["pdf_report"].set_fill_color(0,150,75)
        st.session_state["pdf_report"].set_draw_color(0,150,75)
        st.session_state["pdf_report"].set_line_width(3)
        st.session_state["pdf_report"].set_text_color(255,255,255)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="MACHINE LEARNING EXPERIMENT SETTINGS",ln=True,align='J',border=True,fill=True)
        st.session_state["pdf_report"].set_text_color(0,0,0)
        st.session_state["pdf_report"].set_line_width(0.1)
        st.session_state["pdf_report"].set_draw_color(30,30,30)
        st.session_state["pdf_report"].set_fill_color(255,255,255)
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
        Do not exceed 200 words.
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
        st.session_state["pdf_report"].set_fill_color(0,150,75)
        st.session_state["pdf_report"].set_draw_color(0,150,75)
        st.session_state["pdf_report"].set_line_width(3)
        st.session_state["pdf_report"].set_text_color(255,255,255)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="MACHINE LEARNING MODELS USED",ln=True,align='J',border=True,fill=True)
        st.session_state["pdf_report"].set_text_color(0,0,0)
        st.session_state["pdf_report"].set_line_width(0.1)
        st.session_state["pdf_report"].set_draw_color(30,30,30)
        st.session_state["pdf_report"].set_fill_color(255,255,255)
        st.session_state["pdf_report"].ln(2.5)
        st.session_state["pdf_report"].set_font("Arial",size=9)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=model_exp_response,ln=True,align="J")
        st.session_state["pdf_report"].ln(5)
        
        st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
        st.session_state["pdf_report"].set_fill_color(0,150,75)
        st.session_state["pdf_report"].set_draw_color(0,150,75)
        st.session_state["pdf_report"].set_line_width(3)
        st.session_state["pdf_report"].set_text_color(255,255,255)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="PERFORMANCE COMPARISON OF MACHINE LEARNING MODELS",ln=True,align='J',border=True,fill=True)
        st.session_state["pdf_report"].set_text_color(0,0,0)
        st.session_state["pdf_report"].set_line_width(0.1)
        st.session_state["pdf_report"].set_draw_color(30,30,30)
        st.session_state["pdf_report"].set_fill_color(255,255,255)
        st.session_state["pdf_report"].ln(2.5)
        
        print_df(compare_df,st.session_state["pdf_report"],st.session_state["print_w"])
        st.session_state["pdf_report"].ln(5)
        
        st.session_state["pdf_report"].set_font("Arial",size=9)
        st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=model_comparison_response,ln=True,align="J")
        st.session_state["pdf_report"].ln(5)



with st.sidebar:
    st.info("Click on the \"Download\" button in the sidebar to download the best performing model(.pkl) and a customized report(.pdf).")