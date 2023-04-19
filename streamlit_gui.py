import streamlit as st

def set_streamlit_gui():
    # streamlit gui setting
    st.title("Personalized PDF Chating Bot")

    st.markdown(
        """ 
            ###  Chat with your PDF files with `Conversational Buffer Memory`      
        """
    )

    st.sidebar.markdown(
        """
            #### Steps:
            1. Upload your pdf file
            2. Enter your secret key for embeddings
            3. Perform Q&A
        """
    )

