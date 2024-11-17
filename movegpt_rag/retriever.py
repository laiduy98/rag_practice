import faiss
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings

def load_faqs(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def create_faiss_index(data_path: str, api_key: str):
    faqs = load_faqs(data_path)
    questions = [item["question"] for item in faqs]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(questions, embeddings)
    return vectorstore

def get_faiss_retriever(data_path: str, api_key: str, threshold=0.8):
    vectorstore = create_faiss_index(data_path, api_key)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["score_threshold"] = threshold  # Set similarity threshold
    return retriever