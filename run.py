import gui
# import streamlit as st
from core import UseTool

def run():
    # Initialize the streamlit GUI
    gui.set_streamlit_gui()     

    # Get pages content and set api
    pages = gui.get_pages()
    api = gui.set_api()

    if pages and api:
        # Use tools for chating with a uploaded PDF
        use_tool = UseTool(api, pages)
        use_tool.chain() 


if __name__ == "__main__":
    run()
