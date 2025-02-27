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

# Initialize Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit app title
st.markdown("<h2 style='text-align: center;'>Aplikasi Rekomendasi Wisata & Q&A PDF</h2>", unsafe_allow_html=True)

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template for PDF Q&A
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

# Prompt template for travel recommendations
travel_prompt = ChatPromptTemplate.from_template(
    """
    Anda adalah asisten yang ahli dalam merekomendasikan tempat wisata.
    Berikan rekomendasi tempat wisata di Indonesia berdasarkan preferensi pengguna.
    Pastikan rekomendasi Anda detail dan informatif.
    Preferensi pengguna: {input}
    """
)

# Function to create vector database from uploaded PDF
def create_vector_db_from_pdf(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name

        # Initialize embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

        # Load PDF and split into chunks
        st.session_state.loader = PyPDFLoader(pdf_file_path)
        st.session_state.text_document_from_pdf = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)

        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(
            st.session_state.text_document_from_pdf)

        # Create vector store
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_document_chunks, st.session_state.embeddings)

# Sidebar for PDF upload
with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])

    if pdf_input_from_user is not None:
        if st.button("Buat Vector Database dari PDF"):
            create_vector_db_from_pdf(pdf_input_from_user)
            st.success("Vector Database untuk PDF ini siap digunakan!")

# Main chat interface
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input for chat
user_input = st.text_input("Masukkan pertanyaan atau minta rekomendasi wisata:")

if st.button('Kirim'):
    if user_input:
        if "vector_store" in st.session_state:
            # If PDF is uploaded, prioritize Q&A from PDF
            document_chain = create_stuff_documents_chain(llm, pdf_prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': user_input})
            answer = response['answer']
        else:
            # If no PDF is uploaded, provide travel recommendations
            response = llm.invoke(travel_prompt.format(input=user_input))
            answer = response.content

        # Append to chat history
        st.session_state.chat_history.append(("Anda", user_input))
        st.session_state.chat_history.append(("Bot", answer))

# Display chat history with custom CSS for bubbles and layout
st.markdown("""
    <style>
        .chat-bubble {
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
            max-width: 70%;
        }
        .chat-user {
            background-color: #DCF8C6;
            text-align: right;
            margin-left: auto;
        }
        .chat-bot {
            background-color: #f1f0f0;
            text-align: left;
            margin-right: auto;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: scroll;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 10px;
            background-color: #fff;
            border-top: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# Create chat container with scrollable history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for speaker, text in st.session_state.chat_history:
    if speaker == "Anda":
        st.markdown(f'<div class="chat-bubble chat-user">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble chat-bot">{text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Display input box at the bottom
st.markdown("""
    <div class="input-container">
        <form action="" method="post">
            <input type="text" id="user-input" placeholder="Tulis pesan..." style="width: 90%; padding: 10px;">
            <button type="submit" style="width: 8%; padding: 10px;">Kirim</button>
        </form>
    </div>
""", unsafe_allow_html=True)
