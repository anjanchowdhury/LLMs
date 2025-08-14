from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    #epo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  not working
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
   pipeline_kwargs =dict (
        temperature = 0.5,
        max_new_tokents = 100
    )
)

model=ChatHuggingFace(llm=llm)
# user = input("Ask your query::  ")
# result = model.invoke(user)
# print(result.content)

st.header("Chat models")
user_input = st.text_input("Ask your query::  ")
if st.button("Chat with llm"):
    result = model.invoke(user_input)
    st.write(result.content)