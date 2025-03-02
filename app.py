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

# Define prompts
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

# Function to create vector database from PDF
def create_vector_db_from_pdf(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name
        
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        loader = PyPDFLoader(pdf_file_path)
        text_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(text_documents)
        
        st.session_state.vector_store = FAISS.from_documents(document_chunks, embeddings)
        st.success("Vector Database untuk PDF ini siap digunakan!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])
    if pdf_input_from_user is not None:
        if st.button("Buat Vector Database dari PDF"):
            create_vector_db_from_pdf(pdf_input_from_user)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("🙂").write(msg["content"])
    else:
        st.chat_message("🗿").write(msg["content"])

# User input area using text_input
user_input = st.text_input("Ketik pesan...")
if st.button("Kirim"):
    if user_input:
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("🙂").write(user_input)

        # Fetch response from Groq API
        try:
            if "vector_store" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, pdf_prompt)
                retriever = st.session_state.vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': user_input})
                answer = response.get('answer', "Maaf, saya tidak dapat menemukan jawaban.")
            else:
                response = llm.invoke(travel_prompt.format(input=user_input))
                answer = response.content if response else "Maaf, saya tidak bisa menjawab pertanyaan Anda."
        except Exception as e:
            answer = f"Terjadi kesalahan: {e}"

        # Append assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("🗿").write(answer)

# Button to copy conversation text
if st.button("Salin Teks"):
    conversation_text = "\n".join(
        f"🙂: {msg['content']}" if msg["role"] == "user" else f"🗿: {msg['content']}"
        for msg in st.session_state.messages
    )

    # JavaScript for copying text to clipboard
    st.components.v1.html(f"""
    <textarea id="conversation-text" style="display:none;">{conversation_text}</textarea>
    <button onclick="copyToClipboard()">Salin Teks</button>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById("conversation-text");
        copyText.style.display = "block";
        copyText.select();
        document.execCommand("copy");
        copyText.style.display = "none";
        alert("Percakapan telah disalin sebagai teks!");
    }}
    </script>
    """, height=0)
