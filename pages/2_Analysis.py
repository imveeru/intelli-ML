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

# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
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

# palm.configure(api_key=config["GOOGLE_PALM_API_KEY"])

def ask_llm(prompt):
    # defaults = {
    #     'model': 'models/text-bison-001',
    #     'temperature': 0.7,
    #     'candidate_count': 1,
    #     'top_k': 40,
    #     'top_p': 0.95,
    #     'max_output_tokens': 1024,
    #     'stop_sequences': [],
    #     'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    # }
    
    # res=palm.generate_text(**defaults, prompt=prompt)
    
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


st.title("Automated Exploratory Data Analysis")

def feature_dist(data):
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    rows=int(math.ceil(len(data.columns)/3))
    #st.write(rows)
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(20, 20))

    # Plot histograms
    for ax, column in zip(axes.flatten(), data.columns):
        #sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(column, fontsize=14)
        ax.tick_params(axis='x', labelrotation=45)

    # Adjust the layout
    fig.tight_layout()
    #plt.show()
    plt.title("Feature distribution")
    plt.savefig("dist_plot.png")
    
    pair_plot=sns.pairplot(data).set(title="Pairplot of features")
    # pp_file=pair_plot.get_figure()
    pair_plot.savefig("pair_plot.png")
    
    features_prompt=f'''
    {data.columns}

    The above given array contains the features of a dataset. Write an explanation for each feature.  
    Write it as a single paragraph in an academic tone. No need to mention all the numerical values.
    Do not exceed 200 words.
    '''
    
    desc_prompt=f'''
    {data.describe().to_string()}

    The above given is the description/descirbed of each feature of a dataset.
    Write a detailed comment based on the given data.  
    Write it as a single paragraph in an academic tone. No need to mention all the numerical values.
    Strictly do not exceed 200 words.
    '''
    
    null_values=data.isnull().sum().to_string()
    null_values=null_values.split('\n')
    null_prompt=f'''
    {null_values}

    The above given is the count of null values in each feature of a dataset.
    Write a detailed comment based on the given data.  
    Write it as a single paragraph in an academic tone. No need to mention all the numerical values.
    Also comment on the consequence of the status of null data in the given dataset.
    Do not exceed 100 words.
    '''
    
    dist_prompt=f'''
    {data.skew().to_string()}
    The above given data presents the skewness of each feature of a dataset.
    Write a detailed comment on the distribution of each feature based on the given data.  
    Write it as a single paragraph in an academic tone. No need to mention all the numerical values.

    Note: The skewness data is obtained using the pandas builtin skew function.
    So consider the right and left skew accordingly.

    Also comment on the consequences of such distribution and skewness in data.
    Do not exceed 250 words.
    '''
    
    features_response=ask_llm(features_prompt)
    desc_response=ask_llm(desc_prompt)
    null_response=ask_llm(null_prompt)
    dist_response=ask_llm(dist_prompt)
    
    st.markdown("#### Feature Description")
    st.write(features_response)
    st.pyplot(pair_plot)
    st.markdown("#### Insights on dataset")
    st.write(desc_response)
    st.markdown("##### Insights on Null Values in the dataset")
    st.write(null_response)
    st.markdown("#### Feature Distribution")
    st.pyplot(fig)
    st.write(dist_response)
    
    # st.session_state["pdf_report"].ln(5)
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="FEATURE DESCRIPTION",align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=features_response,align="J")
    # st.session_state["pdf_report"].ln(5)
    
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="INSIGHTS ON DATASET",ln=True,align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=desc_response,ln=True,align="J")
    # st.session_state["pdf_report"].ln(5)
    # st.session_state["pdf_report"].image("pair_plot.png",h=st.session_state["print_w"]/2)
    # st.session_state["pdf_report"].ln(5)
    
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="INSIGHTS ON NULL DATA",ln=True,align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=null_response,ln=True,align="J")
    # st.session_state["pdf_report"].ln(5)
    
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="FEATURE DISTRIBUTION",ln=True,align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=dist_response,ln=True,align="J")
    # st.session_state["pdf_report"].ln(5)
    # st.session_state["pdf_report"].image("dist_plot.png",w=st.session_state["print_w"])
    # st.session_state["pdf_report"].ln(5)
    
def outlier_plot(data):
    # Create a figure and a set of subplots
    rows=int(math.ceil(len(data.columns)/3))
    
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(20, 20))

    # Plot boxplots
    for ax, column in zip(axes.flatten(), data.columns):
        sns.boxplot(y = data[column], ax=ax)
        ax.set_title(column, fontsize=14)

    # Adjust the layout
    fig.tight_layout()
    #plt.show()
    plt.savefig("outlier_plot.png")
    
    outlier_prompt=f'''
    {data.describe().to_string()}

    The above given is the description/descirbed of each feature of a dataset.
    Write a detailed comment only on the outliers in each feature based on the given data.
    Also comment on the presence outliers in higher end and lower end of the scale.
    Do not be too sensitive while commenting. Small variations in outliers can be considered.
    Write it as a paragraph in an academic tone. No need to mention all the numerical values.
    Also include the consequences of such outliers.
    Only comment on impactful features.
    Strictly do not exceed 150 words.
    '''
    
    outlier_response=ask_llm(outlier_prompt)
    
    st.markdown("### Outlier Detection")
    st.pyplot(fig)
    st.write(outlier_response)
    
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="OUTLIER DETECTION",ln=True,align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=outlier_response,ln=True,align="J")
    # st.session_state["pdf_report"].ln(5)
    # st.session_state["pdf_report"].image("outlier_plot.png",w=st.session_state["print_w"])
    # st.session_state["pdf_report"].ln(5)
    
def correlation_plot(data):
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.savefig("corr_plot.png")
    
    correlation_prompt=f'''
    {np.array(data.columns)}

    The above given array is the list of features in a dataset.
    
    {data.corr()}
    
    This is the correlation matrix of the dataset.

    Write a detailed inference of the given correlation matrix by using the above given feature names.
    Give it as a detailed single paragraph in academic tone.
    Do not exceed 100 words
    '''
    
    correlation_response=ask_llm(correlation_prompt)

    st.markdown("### Correlation between features")
    st.pyplot(f)
    st.write(correlation_response)
    
    # st.session_state["pdf_report"].set_font("Arial",size=12,style="B")
    # st.session_state["pdf_report"].set_fill_color(0,150,75)
    # st.session_state["pdf_report"].set_draw_color(0,150,75)
    # st.session_state["pdf_report"].set_line_width(3)
    # st.session_state["pdf_report"].set_text_color(255,255,255)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt="CORRELATION BETWEEN FEATURES",ln=True,align='J',border=True,fill=True)
    # st.session_state["pdf_report"].set_text_color(0,0,0)
    # st.session_state["pdf_report"].set_line_width(0.1)
    # st.session_state["pdf_report"].set_draw_color(30,30,30)
    # st.session_state["pdf_report"].set_fill_color(255,255,255)
    # st.session_state["pdf_report"].ln(2.5)
    # st.session_state["pdf_report"].set_font("Arial",size=9)
    # st.session_state["pdf_report"].multi_cell(w=st.session_state["print_w"],txt=correlation_response,ln=True,align="J")
    # st.session_state["pdf_report"].ln(5)
    # st.session_state["pdf_report"].image("corr_plot.png",w=st.session_state["print_w"])
    # st.session_state["pdf_report"].ln(5)
    

try:
    src=st.session_state["source_data"]
    target=st.selectbox("Select the target variable",src.columns)
    st.session_state["target"] = target
    
    if st.button("Start Analysis"):
        if src.empty:
            st.error("Oops! the uploaded dataset is empty. Kindly reupload or check it.")
        
        if target:
            with st.spinner("Analysing feature distribution..."):
                feature_dist(src)
            
            with st.spinner("Finding outliers..."):
                outlier_plot(src)
            
            with st.spinner("Analysing the correlation between features..."):
                correlation_plot(src)
            
            st.caption("Now click on the modelling button to train and test your dataset in various machine learning models.")
    
except Exception as error:
    st.error("Upload your dataset before staring analysis!")
    st.error(error)



with st.sidebar:
    st.info("Click on the \"Modelling\" button in the sidebar to train and test the uploaded dataset in multiple machine learning models.")
# profile = ProfileReport(src,explorative=True)
# st_profile_report(profile) 