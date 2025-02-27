import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Streamlit UI settings
st.set_page_config(layout="wide")
st.title("🤖 Chatbot Wisata & PDF Q&A")
st.caption("Tanyakan rekomendasi wisata atau ajukan pertanyaan berdasarkan PDF yang diunggah.")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya siap membantu Anda. Tanyakan apa saja!"}]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("🤖"):
            st.write(msg["content"])
    elif msg["role"] == "user":
        with st.chat_message("🙂"):
            st.write(msg["content"])

# User input field
if user_input := st.chat_input("Ketik pertanyaan atau permintaan rekomendasi..."):
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("🙂"):
        st.write(user_input)
    
    # Generate bot response
    response = llm.invoke(user_input)
    bot_reply = response.content
    
    # Append bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("🤖"):
        st.write(bot_reply)
