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

# Streamlit app title
st.markdown("<h2 style='text-align: center;'>Aplikasi Rekomendasi Wisata & Q&A PDF</h2>", unsafe_allow_html=True)

# Initialize Groq LLM
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
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name
        
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        st.session_state.loader = PyPDFLoader(pdf_file_path)
        st.session_state.text_document_from_pdf = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)
        
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)

# Sidebar untuk unggah PDF
with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])
    if pdf_input_from_user is not None:
        if st.button("Buat Vector Database dari PDF"):
            create_vector_db_from_pdf(pdf_input_from_user)
            st.success("Vector Database untuk PDF ini siap digunakan!")

# Inisialisasi chat history di session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Styling CSS agar chat terscroll otomatis dan input tetap di bawah
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column-reverse; /* Membalik urutan chat agar input tetap di bawah */
    overflow-y: auto;
    height: 400px; /* Batasi tinggi chat agar tetap rapi */
    border: 1px solid #ddd;
    padding: 10px;
    margin-bottom: 10px;
}
.chat-bubble-user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    display: inline-block;
    margin: 5px;
    max-width: 70%;
    align-self: flex-end;
}
.chat-bubble-bot {
    background-color: #EAEAEA;
    padding: 10px;
    border-radius: 10px;
    display: inline-block;
    margin: 5px;
    max-width: 70%;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

# Container chat agar input selalu di bawah
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, text in reversed(st.session_state.chat_history):  # Dibalik agar yang terbaru di bawah
        if speaker == "Anda":
            st.markdown(f"<div class='chat-bubble-user'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'>{text}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input pengguna
user_input = st.chat_input("Ketik pesan...")

# Jika ada input, proses dan tambahkan ke chat history
if user_input:
    if "vector_store" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, pdf_prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_input})
        answer = response['answer']
    else:
        response = llm.invoke(travel_prompt.format(input=user_input))
        answer = response.content

    # Tambahkan ke chat history
    st.session_state.chat_history.append(("Anda", user_input))
    st.session_state.chat_history.append(("Bot", answer))

    # Tampilkan kembali halaman agar chat terbaru muncul tanpa menghapus yang lama
    st.experimental_rerun()
