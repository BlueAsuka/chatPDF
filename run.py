import streamlit as st
import streamlit_gui
import util


if __name__ == "__main__":
    # Initialize the streamlit GUI
    streamlit_gui.set_streamlit_gui()

    # Allow user to upload the file
    uploaded_file = st.file_uploader("Please Upload Your PDF File", type=["pdf"])

    # With the upload file, user can read the text chunk on the website
    if uploaded_file:
        name = uploaded_file.name
        text = util.pdf_parser(uploaded_file)
        pages = util.page_chunker(text)

        if pages:
            # The expander to view the text chunk
            with st.expander("Show Page Content", expanded=False): 
                page_sel = st.number_input(
                    label="Select Page",
                    min_value=1,
                    max_value=len(pages),
                    step=1
                )
                pages[page_sel - 1]
