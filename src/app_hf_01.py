import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
import openai
import time
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
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

def create_vector_embedding():
    """Creates vector embeddings and stores them in session state."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = DirectoryLoader("./maintenance")  # Or use file upload
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("Predictive Maintenance")
st.markdown("""
Use this tool to analyze sensor anomalies and identify potential maintenance issues.
1. Input a query or upload relevant documents.
2. Click 'Document Embedding' to prepare the database.
3. Get precise answers and view related documents!
""")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Select Model", ["all-MiniLM-L6-v2", "other-model"])
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)

col1, col2 = st.columns(2)

with col1:
    user_prompt = st.text_input("Enter Anomaly Prediction Query")
    if st.button("Document Embedding"):
        with st.spinner('Creating vector embeddings...'):
            create_vector_embedding()
            st.success("Vector Database is ready!")

with col2:
    if user_prompt and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        try:
            response = retrieval_chain.invoke({'input': user_prompt})
            response_time = time.process_time() - start

            st.subheader("Response")
            st.write(response['answer'])

            st.metric(label="Response Time", value=f"{response_time:.2f} seconds")

        except Exception as e:
            st.error(f"Error: {e}")

        with st.expander("Document Similarity Search"):
            st.subheader("Similar Documents")
            for i, doc in enumerate(response['context']):
                st.write(f"Document {i + 1}:")
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.info("Enter a query and ensure the vector database is ready.")

uploaded_file = st.file_uploader("Upload a document for analysis", type=["pdf", "txt"])
if uploaded_file:
    st.session_state.loader = PyPDFDirectoryLoader(uploaded_file)
