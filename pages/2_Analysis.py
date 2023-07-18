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
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import google.generativeai as palm
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
from dotenv import dotenv_values

config = dotenv_values(".env") 

palm.configure(api_key=config["GOOGLE_PALM_API_KEY"])

def ask_llm(prompt):
    defaults = {
        'model': 'models/text-bison-001',
        'temperature': 0.6,
        'candidate_count': 1,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 1024,
        'stop_sequences': [],
        'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    }
    
    res=palm.generate_text(**defaults, prompt=prompt)
    
    return res.result


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
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(column, fontsize=14)
        ax.tick_params(axis='x', labelrotation=45)

    # Adjust the layout
    fig.tight_layout()
    #plt.show()
    st.markdown("### Feature distribution")
    st.pyplot(fig)
    
    features_prompt=f'''
    {data.columns}

    The above given array contains the features of a dataset. Write an explanation for each feature.  
    Write it as a single paragraph in an academic tone. No need to mention all the numerical values.
    '''
    
    features_response=ask_llm(features_prompt)
    st.write(features_response)
    
    

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
    st.markdown("### Outlier Detection")
    st.pyplot(fig)

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

    st.markdown("### Correlation between features")
    st.pyplot(f)

try:
    src=st.session_state["source_data"]
    target=st.selectbox("Select the target variable",src.columns)
    st.session_state["target"] = target
    
    if src.empty:
        st.error("Oops! the uploaded dataset is empty. Kindly reupload or check it.")
    
    if target:
        with st.spinner("Analysing feature distribution..."):
            feature_dist(src)
        
        with st.spinner("Finding outliers..."):
            outlier_plot(src)
        
        with st.spinner("Analysing the correlation between features..."):
            correlation_plot(src)
    
except Exception as error:
    st.error("Upload your dataset before staring analysis!")
    st.error(error)


# profile = ProfileReport(src,explorative=True)
# st_profile_report(profile) 