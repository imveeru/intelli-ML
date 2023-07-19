import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import *

#page config
st.set_page_config(
    page_title="IntelliML",
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

# st.title("🤖 IntelliML")

with st.sidebar:
    st.info("Click on the \"Upload\" button in the sidebar to upload your dataset.")

st.image("./assets/cover img.jpg")

desc='''
IntelliML is an innovative and powerful machine learning application developed using Python, leveraging the cutting-edge Google's PALM API. Designed to simplify and accelerate the machine learning process, IntelliML empowers users with the ability to upload their datasets, conveniently select the target variable, and obtain valuable insights while effortlessly trainin.get() multiple machine learning models to find the optimal solution for their data.

**Key Features:**

1. **Effortless Dataset Analysis:** With IntelliML, users can easily upload their datasets in various formats. The app automatically performs an in-depth analysis of the data, identifying patterns, distributions, missing values, and correlations. This preliminary analysis provides users with a comprehensive understanding of their data's characteristics.

2. **Target Variable Selection:** Users can conveniently specify the target variable for their analysis. IntelliML smartly recognizes the dependent variable, which is essential for accurate model training.

3. **Automated Machine Learning:** Leveraging the power of Google's PALM API, IntelliML streamlines the machine learning process by automatically training and evaluating a wide range of machine learning models on the uploaded dataset. This includes popular algorithms like Linear Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), Gradient Boosting, Neural Networks, and more.

4. **Model Comparison:** After the automated training process, IntelliML presents users with a comprehensive comparison of the performance of the various trained models. This comparison helps users make informed decisions about selecting the most appropriate model for their specific dataset.

5. **Best Model Recommendation:** IntelliML intelligently identifies the most suitable machine learning model for the uploaded dataset based on its evaluation metrics, such as accuracy, precision, recall, F1-score, and more. This recommendation ensures that users are equipped with the best possible model to make predictions on their data.

6. **Detailed Report Generation:** Once the analysis and model training are complete, IntelliML generates a detailed and user-friendly report summarizing the entire process. This report includes visualizations, insights, evaluation metrics, and other relevant information to aid in understanding the data and model performance better.

**Why Choose IntelliML:**

- **User-Friendly Interface:** IntelliML is designed with a user-friendly interface, making it accessible to both machine learning novices and experts.

- **Time-Efficient:** By automating the time-consuming tasks of dataset analysis and model training, IntelliML significantly reduces the time needed to build effective machine learning models.

- **Data-Driven Insights:** The app provides valuable insights into the uploaded data, enabling users to make data-driven decisions for their business, research, or projects.

- **State-of-the-Art Model Selection:** IntelliML employs the latest advancements in machine learning technology through Google's PALM API, ensuring the detailed explanations for the workflow.

- **Comprehensive Reports:** Users receive detailed and easy-to-understand reports that can be shared with stakeholders or incorporated into presentations.

Developed by [Veeramanohar](https://github.com/imveeru)
'''

st.write(desc,markdown=True)

st.info("As of now, only Regression tasks are supported. Classification, and Clustering tasks will be added soon!",icon="⌛")

# with st.sidebar:
#     st.title("🤖 IntelliML")
#     choice=st.radio("Navigation",["Upload","Analysis","Modelling","Download"])
#     st.info("This application helps you to build an automated ML pipeline.")

# if os.path.exists("sourcedata.csv"):
#     print("Sourcedata")
#     src=pd.read_csv("sourcedata.csv",index_col=None)  

# if choice == "Upload":
#     st.title("Upload your dataset for modelling!")
#     file=st.file_uploader("Upload your CSV file here.")
#     if file:
#         df=pd.read_csv(file,index_col=None)
#         df.to_csv("sourcedata.csv",index=None)
#         st.dataframe(df)
    
#         if st.button("Next >",key="p1"):
#             choice="Analysis"
 
# if choice=="Analysis":
#     st.title("Automated Exploratory Data Analysis")
#     profile = ProfileReport(src,explorative=True)
#     st_profile_report(profile)
    
#     if st.button("Next >",key="p21"):
#         choice="Modelling"
    
#     if st.button("< Previous",key="p22"):
#         choice="Upload"

# if choice=="Modelling":
#     st.title("Modelling")
#     target=st.selectbox("Select the target variable",src.columns)
#     if st.button("Train the model"):
#         setup(src,target=target)
#         setup_df=pull()
#         st.info("Experiment Settings")
#         st.dataframe(setup_df)
#         best_model=compare_models()
#         compare_df=pull()
#         st.info("Model Comparison")
#         st.dataframe(compare_df)
#         save_model(best_model,"best_model")
#         #plot_model(best_model,plot='feature')
        
#     if st.button("Next >",key="p31"):
#         choice="Download"

#     if st.button("< Previous",key="p32  "):
#         choice="Analysis"

# if choice=="Download":
#     with open("best_model.pkl",'rb') as f:
#         st.download_button("Download the best model",f,"best_model.pkl")