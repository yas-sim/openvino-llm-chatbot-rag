#test-rag-client.py
# 

import json
import requests
import streamlit as st

st.title('OpenVINO Q&A Chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Your input here.')

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})

    with st.chat_message('assistant'):
        payload = {"query":prompt}
        ans = requests.get('http://127.0.0.1:8000/chatbot/1', params=payload)
        ans = json.loads(ans.text)
        st.markdown(ans['response'])
        st.session_state.messages.append({'role':'assistant', 'content':ans['response']})

# http://127.0.0.1:8000/docs

# streamlit run test-rag-client.py

