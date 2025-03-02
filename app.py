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
st.title("Aplikasi Rekomendasi Wisata & Q&A PDF")
with st.expander("ℹ️ Disclaimer"):
    st.caption(
        """We appreciate your engagement! Please note, this demo is designed to
        process a maximum of 10 interactions and may be unavailable if too many
        people use the service concurrently. Thank you for your understanding.
        """
    )

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
        
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        loader = PyPDFLoader(pdf_file_path)
        text_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(text_documents)
        
        st.session_state.vector_store = FAISS.from_documents(document_chunks, embeddings)
        st.success("Vector Database untuk PDF ini siap digunakan!")

with st.sidebar:
    st.header("Unggah PDF")
    pdf_input_from_user = st.file_uploader("Unggah file PDF", type=['pdf'])
    if pdf_input_from_user is not None:
        if st.button("Buat Vector Database dari PDF"):
            create_vector_db_from_pdf(pdf_input_from_user)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if len(st.session_state.chat_history) >= 10:
    st.info(
        """Notice: The maximum message limit for this demo version has been reached. We value your interest!
        We encourage you to experience further interactions by building your own application with instructions
        from Streamlit's [Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
        tutorial. Thank you for your understanding."""
    )
else:
    chat_container = st.container()
    with chat_container:
        for speaker, text in st.session_state.chat_history:
            alignment = "flex-end" if speaker == "Anda" else "flex-start"
            bg_color = "#DCF8C6" if speaker == "Anda" else "#EAEAEA"
            st.markdown(
                f"""
                <div style='display: flex; flex-direction: column; align-items: {alignment};'>
                    <div style='background-color: {bg_color}; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;'>
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Input text box placed outside the chat container
    user_input = st.text_input("Ketik pesan...")
    if st.button("Kirim"):
        if user_input:
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
            
            st.session_state.chat_history.append(("Anda", user_input))
            st.session_state.chat_history.append(("Bot", answer))
            st.experimental_rerun()
