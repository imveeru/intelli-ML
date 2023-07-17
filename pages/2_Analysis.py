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
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

st.title("Automated Exploratory Data Analysis")

def feature_dist(data):
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    rows=math.ceil(len(data.columns)/3)
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
    st.pyplot(fig)

def outlier_plot(data):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))

    # Plot boxplots
    for ax, column in zip(axes.flatten(), data.columns):
        sns.boxplot(y = data[column], ax=ax)
        ax.set_title(column, fontsize=14)

    # Adjust the layout
    fig.tight_layout()
    #plt.show()
    st.pyplot(fig)


try:
    src=st.session_state["source_data"]
    if src.empty:
        st.error("Upload your dataset before staring analysis!")
    
    corr=src.corr()
    heatmap=sns.heatmap(corr)
    
    feature_dist(src)
    outlier_plot(src)
    
except Exception:
    st.error("Upload your dataset before staring analysis!")




# profile = ProfileReport(src,explorative=True)
# st_profile_report(profile) 