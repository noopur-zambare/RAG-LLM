import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

if 'theme' not in st.session_state:
    st.session_state.theme = "day"

def toggle_theme():
    st.session_state.theme = "night" if st.session_state.theme == "day" else "day"

st.markdown("""
    <style>
    .theme-toggle {
        position: fixed;
        top: 0px !important;
        right: 1000px !important;
        z-index: 1000;
    }
    .stButton button {
        min-width: 40px;
        padding: 5px 10px;
            margin: 0 !important;
        
    }
    </style>
    """, unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    button_label = "🌙" if st.session_state.theme == "day" else "☀️"
    st.button(button_label, on_click=toggle_theme, key="theme_button")
    st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.theme == "night":
    st.markdown("""
        <style>
        .theme-toggle button {
        background: #FFFFFF !important;}
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .stChatInput input {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border: 1px solid #3A3A3A !important;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #1E1E1E !important;
            border: 1px solid #3A3A3A !important;
            color: #E0E0E0 !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #2A2A2A !important;
            border: 1px solid #404040 !important;
            color: #F0F0F0 !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .stChatMessage .avatar {
            background-color: #00FFAA !important;
            color: #000000 !important;
        }
        .stChatMessage p, .stChatMessage div {
            color: #FFFFFF !important;
        }
        .stFileUploader {
            background-color: #1E1E1E;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            padding: 15px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FFAA !important;
        }
        .stSelectbox select:focus {
            outline: none;
            border-color: #2E7D32;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .theme-toggle button {
        background: #FFFFFF !important;}
                
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stChatInput input {
            background-color: #F1F1F1 !important;
            color: #000000 !important;
            border: 1px solid #CCCCCC !important;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #F1F1F1 !important;
            border: 1px solid #CCCCCC !important;
            color: #000000 !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #E0E0E0 !important;
            border: 1px solid #CCCCCC !important;
            color: #000000 !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .stChatMessage .avatar {
            background-color: #4CAF50 !important;
            color: #000000 !important;
        }
        .stChatMessage p, .stChatMessage div {
            color: #000000 !important;
        }
        .stFileUploader {
            background-color: #F1F1F1;
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            padding: 15px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #4CAF50 !important;
        }
        .stSelectbox select:focus {
            outline: none;
            border-color: #2E7D32;
        }
        </style>
        """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'data/'


st.title("RAG-LLM")
st.markdown("### Chat with your document")
st.markdown("---")
st.markdown("###### Select a Model")
embedding_model_choice = st.selectbox(
    label='',
    options=["deepseek-r1:1.5b", "llama3.3", "mixtral"],
    index=0,
    help="Choose the model for document analysis."
)


if embedding_model_choice == "deepseek-r1:1.5b":
    EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
    LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
elif embedding_model_choice == "llama3.3":
    EMBEDDING_MODEL = OllamaEmbeddings(model="llama3.3")
    LANGUAGE_MODEL = OllamaLLM(model="llama3.3")
elif embedding_model_choice == "mixtral":
    EMBEDDING_MODEL = OllamaEmbeddings(model="mixtral")
    LANGUAGE_MODEL = OllamaLLM(model="mixtral")

DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)