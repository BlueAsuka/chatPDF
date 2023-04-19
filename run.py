import gui
import core


def run():
    # Initialize the streamlit GUI
    gui.set_streamlit_gui()

    pages = gui.get_pages()

    # Retrieval Q&A and use tools
    index = core.embed(pages)
    retrieval = core.retrieval(index)


if __name__ == "__main__":
    run()
