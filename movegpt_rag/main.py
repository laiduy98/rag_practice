from fastapi import FastAPI, HTTPException
from movegpt_rag.config import settings
from movegpt_rag.config import DATA_PATH, SIMILARITY_THRESHOLD
from movegpt_rag.retriever import get_faiss_retriever
from movegpt_rag.chat import create_qa_chain

app = FastAPI()


retriever = get_faiss_retriever(DATA_PATH, settings.openai_api_key, SIMILARITY_THRESHOLD)
qa_chain = create_qa_chain(retriever, settings.openai_api_key)

@app.get("/")
def read_root():
    return {"message": "Footura Healthcare RAG API test"}

@app.post("/ask_movegpt/")
def ask_question(question: str):
    try:
        # Retrieve relevant FAQs
        docs = retriever.get_relevant_documents(question)
        
        # Check if relevant FAQ is found
        if not docs:
            return {"question": question, "answer": "I'm sorry, I couldn't find any relevant information."}

        # Context for ChatGPT
        context = docs[0].page_content
        response = qa_chain({"context": context, "question": question})

        # Extract the answer
        answer = response["result"]
        return {"question": question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))