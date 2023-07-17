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
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

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
    if src.empty:
        st.error("Oops! the uploaded dataset is empty. Kindly reupload or check it.")
    
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