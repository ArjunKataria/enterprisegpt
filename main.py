import openai
import pandas as pd
import numpy as np
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from streamlit_chat import message

# This is a long document we can split up.
with open('f.txt') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 650,
    chunk_overlap  = 50,
    length_function = len,
)


texts = text_splitter.create_documents([state_of_the_union])

os.environ["OPENAI_API_KEY"] = "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"

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database (unhash db = FAISS & db.save_local when uplaoding new files)
#db = FAISS.from_documents(texts, embeddings)

#db.save_local("faiss_indexall_product")


#load the vector store
new_db = FAISS.load_local("faiss_indexall_product", embeddings)


from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. Always give answer step by step with detailed short summary of the question with there pros and cons in new line.

{context}

Question: {question}

FINAL ANSWER

Answer:

'Answer step by step with  detailed  summary in minimum 350 words  and maximum 550 words of the question  with there pros and cons in new line with headers '
'and give techjocjkey links for relevant question saying you can visit techjockey for comparing'


"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


chain_type_kwargs = {"prompt": PROMPT}
tp = OpenAI(model_name = "text-davinci-003",temperature=0.4,max_tokens=1200)
qa = RetrievalQA.from_chain_type(llm=tp,
                                 chain_type="stuff", retriever=new_db.as_retriever(search_type="mmr", search_kwargs={"score_threshold": .3}), chain_type_kwargs=chain_type_kwargs,verbose ='True')

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# chat completion llm
llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301',temperature=0.1,max_tokens=200)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=2,
    return_messages=True
)

from langchain.agents import Tool

tools = [
    Tool(
        name='Knowledge_Base',
        func=qa.run,
        description=('use this tool when answering question about firewall or cyber security or networking or Authentication'),
        return_direct=True

    )
]

from langchain.agents import initialize_agent

agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=tp,
    stop=["\nObservation:"],
    max_iterations=1,
    memory=conversational_memory,
    early_stopping_method='generate',
    verbose = 'True',
    handle_parsing_errors="Check your output and make sure it confirms!"
)

def chatbot():
    query = st.text_input("Enter your question:")
    if query:
        result = agent.run(query)
        chat_history = []
        st.write("User question:", query)
        st.write("Chatbot answer:", result)
        # Initialize the chat history.
        # Add the current chat to the chat history.
        chat_history.append((query, result))
        # Set the chat history as the session state.
        st.session_state.chat_history = chat_history
        # Get the previous chat history.
        previous_chat_history = st.session_state.chat_history
        # Display the previous chat history.



# Create a Streamlit app
def app():
    # Set the title of the app
    st.title("Enterprise Chatbot")
    # Add a description of the app
    st.write("Ask me anything!")
    # Call the chatbot function
    chatbot()

# Run the app
if __name__ == "__main__":
    app()
