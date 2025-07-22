from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

@st.cache_resource(show_spinner=True)
def load_model():
    with st.spinner("Downloading model from Hugging Face... This may take a while ⏳"):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

def get_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
