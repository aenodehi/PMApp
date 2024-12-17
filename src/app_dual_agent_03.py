import streamlit as st
import os
import subprocess
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

# Initialize the model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompts for Explorer, Evaluator, and the new agent
explorer_prompt = ChatPromptTemplate.from_template(
    """
    Explore the topic based on the provided context.
    Provide a detailed and comprehensive explanation to the query.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

evaluator_prompt = ChatPromptTemplate.from_template(
    """
    Critically evaluate the provided context and the query.
    Provide an analysis that highlights strengths, weaknesses, or key considerations.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# New agent (e.g., a "Summary Agent") prompt
summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the provided context and answer the query with a concise summary.
    Focus on key points and insights.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Vector embedding setup
def create_vector_embedding():
    """Creates vector embeddings and stores them in session state."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = DirectoryLoader("./maintenance") 
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# Streamlit interface
st.set_page_config(page_title="Predictive Maintenance (Dual Agent)", layout="wide")

st.title("Predictive Maintenance (Dual Agent)")
st.markdown("""
Analyze anomalies and receive insights from three different perspectives:
1. **Explorer Agent**: Provides a broad, exploratory response.
2. **Evaluator Agent**: Offers critical analysis and evaluation.
3. **Summary Agent**: Summarizes the key points and insights.
""")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Select Model", ["all-MiniLM-L6-v2", "other-model"])
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)

st.sidebar.image(
    "/home/ali/Projects/LLMs/RAG_Document_QA_GROQ_API_LLama3_20241102/images/02.png",
    caption="Predictive Maintenance",
    use_column_width=True
)

col1, col2 = st.columns(2)

with col1:
    user_prompt = st.text_input("Enter Anomaly Prediction Query")
    if st.button("Document Embedding"):
        with st.spinner('Creating vector embeddings...'):
            create_vector_embedding()
            st.success("Vector Database is ready!")

# Simulated browser-style tabs using `st.columns` for tab navigation
tab1, tab2, tab3 = st.columns([1, 1, 1])

with tab1:
    selected_tab_1 = st.button("Explorer Agent", key="tab1")
with tab2:
    selected_tab_2 = st.button("Evaluator Agent", key="tab2")
with tab3:
    selected_tab_3 = st.button("Summary Agent", key="tab3")

# Handle the tab click state
if selected_tab_1:
    selected_tab = "Explorer Agent"
elif selected_tab_2:
    selected_tab = "Evaluator Agent"
elif selected_tab_3:
    selected_tab = "Summary Agent"
else:
    selected_tab = "Explorer Agent"  # Default to Explorer Agent

# Agent handling based on selected tab
if user_prompt and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()

    # Define the chains for each agent
    explorer_chain = create_stuff_documents_chain(llm, explorer_prompt)
    explorer_retrieval_chain = create_retrieval_chain(retriever, explorer_chain)

    evaluator_chain = create_stuff_documents_chain(llm, evaluator_prompt)
    evaluator_retrieval_chain = create_retrieval_chain(retriever, evaluator_chain)

    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    summary_retrieval_chain = create_retrieval_chain(retriever, summary_chain)

    start = time.process_time()

    try:
        # Get responses for each agent
        explorer_response = explorer_retrieval_chain.invoke({'input': user_prompt})
        evaluator_response = evaluator_retrieval_chain.invoke({'input': user_prompt})
        summary_response = summary_retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start

        # Show the content of the selected tab
        if selected_tab == "Explorer Agent":
            st.subheader("Explorer Agent Response")
            st.write(explorer_response['answer'])
        elif selected_tab == "Evaluator Agent":
            st.subheader("Evaluator Agent Response")
            st.write(evaluator_response['answer'])
        elif selected_tab == "Summary Agent":
            st.subheader("Summary Agent Response")
            st.write(summary_response['answer'])

        st.metric(label="Response Time", value=f"{response_time:.2f} seconds")

    except Exception as e:
        st.error(f"Error: {e}")

    # Show document similarity results
    with st.expander("Document Similarity Search"):
        st.subheader("Similar Documents")
        for i, doc in enumerate(explorer_response['context']):
            st.write(f"Document {i + 1}:")
            st.write(doc.page_content)
            st.write('------------------------')
else:
    st.info("Enter a query and ensure the vector database is ready.")

# File upload
uploaded_file = st.file_uploader("Upload a document for analysis", type=["pdf", "txt"])
if uploaded_file:
    st.session_state.loader = PyPDFDirectoryLoader(uploaded_file)
