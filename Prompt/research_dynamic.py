from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
    )

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematics"])

length_input = st.selectbox("Select Explanation Length", ["short (1-2 paragraph)", "Medium (3-5 paragraph)", "Long (detailed explanation)"])

#Template
template = PromptTemplate(
    template = """
Please summarize the research paper tited "{paper_input}" with the following specifications: Explanation Style: {style_input} Explanation Length:{length_input}
1. Mathematical Details:
    -Include relevant mathematical equations if present in the paper.
    -Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
    -Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables = ['paper_input', 'style_input', 'length_input'],
validate_template = True
)

# Fill the placeholders
prompt = template.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
