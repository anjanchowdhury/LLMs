from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
   pipeline_kwargs =dict (
        temperature = 0.5,
        max_new_tokents = 100
    )
)
model = ChatHuggingFace(llm=llm)

st.header("Srearch Tool")
user_input=st.text_input("Enter your prompt: ")
if st.button("Summerize"):
    result = model.invoke(user_input)
    st.write(result.content)