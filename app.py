import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Streamlit App
st.set_page_config(layout="wide")
st.title("💬 Chatbot Rekomendasi Wisata & Q&A PDF")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya siap membantu Anda. Ajukan pertanyaan atau minta rekomendasi wisata."}
    ]

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])
    
    if pdf_input_from_user is not None and st.button("Buat Vector Database dari PDF"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_input_from_user.read())
            pdf_file_path = temp_file.name
        
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        loader = PyPDFLoader(pdf_file_path)
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(text_documents)
        
        st.session_state.vector_store = FAISS.from_documents(document_chunks, embeddings)
        st.success("Vector Database untuk PDF ini siap digunakan!")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("🤖").write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message("🙂").write(msg["content"])

# Handle user input
user_input = st.chat_input("Ketik pertanyaan atau minta rekomendasi...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("🙂").write(user_input)
    
    if "vector_store" in st.session_state:
        retriever = st.session_state.vector_store.as_retriever()
        document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
            Jawab pertanyaan berdasarkan konteks berikut:
            <context>
            {context}
            </context>
            Pertanyaan: {input}
        """))
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_input})
        answer = response['answer']
    else:
        response = llm.invoke(ChatPromptTemplate.from_template("""
            Anda adalah asisten yang ahli dalam merekomendasikan tempat wisata di Indonesia.
            Berikan rekomendasi berdasarkan preferensi berikut:
            {input}
        """).format(input=user_input))
        answer = response.content
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("🤖").write(answer)
