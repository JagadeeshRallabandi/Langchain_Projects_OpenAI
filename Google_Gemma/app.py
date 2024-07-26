import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the environment variables are loaded correctly
if groq_api_key is None:
    st.error("GROQ_API_KEY is not set. Please check your .env file.")
if google_api_key is None:
    st.error("GOOGLE_API_KEY is not set. Please check your .env file.")

# Set the GOOGLE_API_KEY environment variable for the Google Generative AI Embeddings
os.environ['GOOGLE_API_KEY'] = google_api_key

# Streamlit app title
st.title("Gemma Model Document Q&A")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize session state variables
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Create an empty FAISS index
        st.session_state.vectors = None

        # Add documents to the FAISS index in batches
        batch_size = 100
        for i in range(0, len(st.session_state.final_documents), batch_size):
            batch = st.session_state.final_documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = st.session_state.embeddings.embed_documents(texts)
            if st.session_state.vectors is None:
                st.session_state.vectors = FAISS.from_texts(texts, st.session_state.embeddings)
            else:
                st.session_state.vectors.add_texts(texts, embeddings)

# Input field for the question
prompt1 = st.text_input("What do you want to ask from the Documents?")

# Button to create the vector store
if st.button("Creating a Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is ready")

# Handle the question input and generate the response
if prompt1:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Vector store is not initialized. Please create the vector store first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------------------------------------------")
