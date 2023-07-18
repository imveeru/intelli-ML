import streamlit as st
st.set_page_config(
    page_title="IntelliML - Uploading",
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

st.title("Download the model and report")
with open("best_model.pkl",'rb') as f:
        st.download_button("Download the best model",f,"best_model.pkl")

st.session_state["pdf_report"].output("final_report.pdf")

with open("final_report.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Download Report",
                    data=PDFbyte,
                    file_name="Final Report by IntelliML.pdf",
                    mime='application/octet-stream')

# with open('sourcedata.csv', 'w') as creating_new_csv_file: 
#    pass 