import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and Embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate respone based on the question
<context>
{context}
<context>
Question: {input}
""")

# ✅ Vector embedding creation with safety checks
def create_vector_embedding():
    # Initialize embedding model in session_state if not already
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load documents only if not already loaded
    if "docs" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()

    # Split documents
    if "final_documents" not in st.session_state or not st.session_state.final_documents:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # ✅ Check final_documents before FAISS
    if not st.session_state.final_documents:
        st.error("❌ No documents found to embed.")
        return

    # ✅ Create FAISS vector store with error handling
    try:
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.success("✅ Vector Database is ready!")
    except Exception as e:
        st.error(f"⚠️ Failed to create vector store: {e}")

# UI Layout
st.title("📄 RAG Document Q&A With Groq And Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

# Trigger embedding
if st.button("📥 Create Document Embedding"):
    create_vector_embedding()

# Handle Q&A
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please embed the documents first by clicking 'Create Document Embedding'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"⏱️ Response time: {round(time.process_time() - start, 2)} seconds")
        st.write("### ✅ Answer:")
        st.write(response['answer'])

        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write('---')
