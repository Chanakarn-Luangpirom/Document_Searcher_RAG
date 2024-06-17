# About streamlit pages: https://www.youtube.com/watch?v=YClmpnpszq8&ab_channel=CodingIsFun
import streamlit as st
import random
import time
from document_searcher import letter_tokenizer, preprocess_text, generate_prompt, rerank_documents, DocSearcher
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import google.generativeai as genai
import pickle
import os
from streamlit_feedback import streamlit_feedback
from googleDriveUpload import upload_json_to_google_drive
from datetime import datetime
import re




st.set_page_config(page_title = 'Multipage App')
st.title("RAG App Document Searcher")

@st.cache_data
def load_documents():
    return pd.read_excel('data/ground_truth.xlsx')

@st.cache_resource
def load_document_searcher(gt):
    return DocSearcher(gt)

@st.cache_resource
def load_llm():
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    return genai.GenerativeModel('gemini-1.5-pro')

gt = load_documents()
d = load_document_searcher(gt)
llm_model = load_llm()


scoring_model = st.sidebar.radio(
        "Choose a scoring model",
        ("Ensemble","TF-IDF", "Google_Embeddings")
    )


rerank_flag = st.sidebar.radio(
    "Would you like the documents to be reranked?",
    ("No","Yes")
)

num_sections = st.sidebar.slider(
    "Input number of sections queried",
    1,3,1
    )

num_questions = st.sidebar.slider(
    "Input number of Q&A queried",
    num_sections,12,3*num_sections
    )

def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    thumb = user_response['score']
    feedback = user_response['text']
    user_input = st.session_state.messages[-2]['content']
    llm_output = st.session_state.messages[-1]['content']
    data = {
    'thumb': thumb,
    'feedback': feedback,
    'user_input': user_input,
    'llm_output': llm_output
    }

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)   
    upload_json_to_google_drive(data,current_time,parent_folder_id='1i307DYh9OFoPC6AkxI1rTwrhC0J3HKO-') # Folder to store feedback




# Streamed response emulator
def response_generator(response):
    for chunk in re.split(r'(\s+)', response):
        yield chunk
        time.sleep(0.01)
        

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

# Display chat messages from history on app rerun
for n,message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and n >= 1:
            feedback_key = f"feedback_{int(n)}"

            streamlit_feedback(
                feedback_type = 'thumbs',
                optional_text_label = 'Please provide extra information',
                on_submit = _submit_feedback,
                key=feedback_key
            )


# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    if scoring_model == 'TF-IDF':
        retrieved_documents = d.query_documents(prompt,k = num_questions, method = 'tfidf', num_section = num_sections)
        if rerank_flag == 'Yes':
            retrieved_documents = rerank_documents(prompt,retrieved_documents)
            output = generate_prompt(prompt,retrieved_documents,d.group_key)
        else:
            output = generate_prompt(prompt,retrieved_documents,d.group_key)

    elif scoring_model == 'Google_Embeddings':
        retrieved_documents = d.query_documents(prompt,k = num_questions, method = 'vector', num_section = num_sections)
        if rerank_flag == 'Yes':
            retrieved_documents = rerank_documents(prompt,retrieved_documents)
            output = generate_prompt(prompt,retrieved_documents,d.group_key)
        else:
            output = generate_prompt(prompt,retrieved_documents,d.group_key)

    elif scoring_model == 'Ensemble':
        retrieved_documents = d.query_documents(prompt,k = num_questions, method = 'all', num_section = num_sections)
        if rerank_flag == 'Yes':
            retrieved_documents = rerank_documents(prompt,retrieved_documents)
            output = generate_prompt(prompt,retrieved_documents,d.group_key)
        else:
            output = generate_prompt(prompt,retrieved_documents,d.group_key)


    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

 
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if output != "Not confident enough to generate prompt":
            response = st.write_stream(response_generator('The Given Context:' + output))
            st.session_state.messages.append({"role": "assistant", "content": response})
            llm_response = llm_model.generate_content(output)
            response = st.write_stream(response_generator(str(llm_response.text)))

        else:
            response = st.write_stream(response_generator(output))



            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()




