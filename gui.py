import streamlit as st
import utils


def set_streamlit_gui():
    """ streamlit gui setting """
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


def get_pages():
    """ Show pages text and get pages text after upload PDF file """
    # Allow user to upload the file
    uploaded_file = st.file_uploader("Please Upload Your PDF File", type=["pdf"])

    # With the upload file, user can read the text chunk on the website
    if uploaded_file:
        name = uploaded_file.name
        text = utils.pdf_parser(uploaded_file)
        pages = utils.page_chunker(text)

        if pages:
            # The expander to view the text chunk
            with st.expander("Show Page Content", expanded=False): 
                page_sel = st.number_input(
                    label="Select Page",
                    min_value=1,
                    max_value=len(pages),
                    step=1
                )
                pages[page_sel - 1]  # Show the page number
    
    return pages


def set_openai_api():
    """ Setting OpenAI API """
    api = st.text_input(
        "Enter OpenAI API Key",
        type='password',
        placeholder="sk-",
        help="https://platform.openai.com/account/api-keys",
    )

    return api
