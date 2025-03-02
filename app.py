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

# Inisialisasi LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt untuk pertanyaan dari PDF
pdf_prompt = ChatPromptTemplate.from_template(
    """
    Jawab pertanyaan berdasarkan konteks berikut:
    <context>
    {context}
    </context>
    Pertanyaan: {input}
    """
)

# Prompt untuk rekomendasi wisata
travel_prompt = ChatPromptTemplate.from_template(
    """
    Anda adalah asisten wisata yang dapat merekomendasikan tempat wisata di Indonesia.
    Gunakan informasi berikut sebagai referensi: {input}
    """
)

# Fungsi untuk membuat database vektor dari PDF
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
            st.success("Vector Database PDF siap!")

# Inisialisasi chat history di session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Tampilkan riwayat chat
st.markdown("## Chat Bot")
for speaker, text in st.session_state.chat_history:
    if speaker == "Anda":
        st.markdown(f"**🧑 Anda:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# Input pengguna **harus berada di luar kondisi if**
user_input = st.chat_input("Ketik pesan...")

# Jika pengguna mengirim pesan
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

    # Simpan chat
    st.session_state.chat_history.append(("Anda", user_input))
    st.session_state.chat_history.append(("Bot", answer))

    # Refresh halaman agar chat terbaru muncul
    st.experimental_rerun()
