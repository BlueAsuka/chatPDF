import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, OpenAI


class UseTool():
    """ Tools and agents for text embedding, retrieval and Q&A"""

    def __init__(self, api, pages) -> None:
        self.api = api
        self.pages = pages


    @st.cache_data
    def embed(_self):
        # embed the pages using LLM
        emb = OpenAIEmbeddings(openai_api_key=_self.api)

        # Indexing and save embedding results in a VDB
        with st.spinner("Embedding..."):
            index = FAISS.from_documents(_self.pages, emb)
        st.success("Embedding done")
    
        return index


    def chain(self):
        if self.api:
            # embed and index
            index = self.embed()

            # use the retrieval-based question-answering (QA) system
            qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=self.api),
                    chain_type="stuff",
                    retriever=index.as_retriever()
                )

            # ensure running the qa chain that just defined
            tools = [
                Tool(
                    name="State of PDF QA System",
                    func=qa.run(),
                    description="Useful for asking personal questions."
                )
            ]

            # Define the prompt that we use the LLM
            prompt_prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                                You have access to a single tool:"""
            prompt_suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prompt_prefix,
                suffix=prompt_suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            # store the entity memory
            if "memoery" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )
            
            # Create a chain of language models that can generate text
            llm_chain = LLMChain(
                    llm=OpenAI(
                        temperature=0,
                        openai_api_key=self.api,
                        model_name="gpt-3.5-turbo"),
                    prompt=prompt
            )

            # Execute the conversational agent
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                            agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            # define the query to give a text input
            query = st.text_input(
                "What is your question?",
                placeholder="Ask me anything from the uploaded file"
            )

            if query:
                res = agent_chain.run(query)
                st.write(res)
            
            with st.expander("History"):
                st.session_state.memory
