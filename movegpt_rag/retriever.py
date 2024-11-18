# import faiss
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

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

def create_faiss_index_sentence_transformer(data_path: str, model_name: str = "all-MiniLM-L6-v2"): # all-distilroberta-v1
    faqs = load_faqs(data_path)
    questions = [item["question"] for item in faqs]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, show_progress_bar=True)
    vectorstore = FAISS.from_texts(questions, embeddings)
    return vectorstore

def get_faiss_retriever_sentence_transformer(data_path: str, model_name: str = "all-MiniLM-L6-v2", threshold=0.8): # all-distilroberta-v1
    vectorstore = create_faiss_index_sentence_transformer(data_path, model_name)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["score_threshold"] = threshold  # Set similarity threshold
    return retriever

# Other method to do the smilarity search
# Annoy, ScaNN, HNSW etc.
#
# Other method to do Embeddings
# SBERT, Roberta ?