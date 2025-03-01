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

st.markdown("<h2 style='text-align: center;'>Aplikasi Rekomendasi Wisata & Q&A PDF</h2>", unsafe_allow_html=True)

if not groq_api_key:
    st.error("API key untuk Groq tidak ditemukan. Silakan tambahkan ke .env file.")
else:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

pdf_prompt = ChatPromptTemplate.from_template(
    """
    Jawab pertanyaan berdasarkan konteks yang diberikan.
    Berikan respons yang paling akurat sesuai dengan pertanyaan.
    <context>
    {context}
    </context>
    Pertanyaan: {input}
    """
)

travel_prompt = ChatPromptTemplate.from_template(
    """
    Anda adalah asisten yang ahli dalam merekomendasikan tempat wisata.
    Berikan rekomendasi tempat wisata di Indonesia berdasarkan preferensi pengguna.
    Pastikan rekomendasi Anda detail dan informatif.
    Preferensi pengguna: {input}
    """
)

def create_vector_db_from_pdf(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
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
        st.success("Vector Database untuk PDF siap digunakan!")

with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])
    if pdf_input_from_user and st.button("Buat Vector Database dari PDF"):
        create_vector_db_from_pdf(pdf_input_from_user)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Masukkan pertanyaan atau minta rekomendasi wisata:", key="user_input")
if st.button('Kirim') and user_input:
    if "vector_store" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, pdf_prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_input})
        answer = response.get('answer', "Maaf, saya tidak dapat menemukan jawaban yang relevan.")
    else:
        response = llm.invoke(travel_prompt.format(input=user_input))
        answer = response.content if hasattr(response, 'content') else "Maaf, saya tidak dapat memberikan rekomendasi."
    
    st.session_state.chat_history.append(("Anda", user_input))
    st.session_state.chat_history.append(("Bot", answer))

st.markdown("""
    <style>
        .chat-bubble { border-radius: 15px; padding: 10px; margin: 5px; max-width: 70%; }
        .chat-user { background-color: #DCF8C6; text-align: right; margin-left: auto; }
        .chat-bot { background-color: #f1f0f0; text-align: left; margin-right: auto; }
        .chat-container { max-height: 500px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for speaker, text in st.session_state.chat_history:
    if speaker == "Anda":
        st.markdown(f'<div class="chat-bubble chat-user">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble chat-bot">{text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
