# app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

st.title("💬 SmartTalk - Mini ChatGPT")
st.markdown("Ask me anything!")

# Chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

user_input = st.text_input("You:", key="chat_input")

if user_input:
    # Encode the user input and append it to the chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and display
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.text_area("SmartTalk:", value=response, height=100, max_chars=None, key="response", disabled=True)

    # Save history
    st.session_state.chat_history_ids = chat_history_ids


