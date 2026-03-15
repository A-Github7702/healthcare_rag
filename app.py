import streamlit as st
from rag import HealthcareRAG

st.title("🩺 Healthcare RAG Chatbot")
st.caption("*For educational use only. Consult a doctor for medical advice.*")

rag = HealthcareRAG()
qa_chain = rag.get_qa_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your symptoms or ask about medications"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data..."):
            response = qa_chain.run(prompt + 
                "\n\nIMPORTANT: Provide general information only. "
                "Advise consulting healthcare professional.")
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})