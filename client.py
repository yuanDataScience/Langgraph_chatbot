import requests
import streamlit as st



st.write("Upload a file to FastAPI")
file = st.file_uploader("Choose a file", type=["pdf"])

if st.button("Submit"):
    if file is not None:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload", files=files)
        st.write(response.text)
    else:
        st.write("No file uploaded.")

st.title("FastAPI Text ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Write your prompt in this input field"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.text(prompt)

    if not st.session_state.messages:
        question = prompt
    else:
        rephased_question = requests.post(
            "http://localhost:8000/rephrase_question",
            json={
                "question": prompt,
                "chat_history": st.session_state.messages
                }
        )
        question = rephased_question.json()        

    response = requests.post(
        f"http://localhost:8000/generate_text", 
        json={"prompt": question}
    )
    response.raise_for_status()
    
    response_json = response.json()
    answer = response_json["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    
   
