from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

def create_qa_chain(retriever, api_key):
    # Custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a healthcare assistant of Footura. Use the following context to answer the question.\n"
            "Context: {context}\n"
            "Question: {question}\n"
            # "If the context does not answer the question, respond with 'I'm sorry, I cannot help with that.'"
        )
    )

    # Define LLM with restrictive prompt
    llm = OpenAI(openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt_template}, input_key="question"
    )
    return qa_chain