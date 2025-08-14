from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Kolkata is capital of WestBengal",
    "Delhi is capital of India",
    "Tokyo is the capital of Japan",
    "Russia's capital is Moscow"
    ]

text = "Tell me about moscow"
docs = model.embed_documents(documents)

query = model.embed_query(text)
result = cosine_similarity([query],docs)[0]
index,scores = sorted(list(enumerate(result)), key = lambda x: x[1])[-1]
print(text)
print(scores)
print(index)
print(documents[index])

print(f"matching score is {scores} similar")

