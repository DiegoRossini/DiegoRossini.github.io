import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
import re

# Load and parse Q&A data from a text file
def load_qa_data(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    qa_pairs = re.findall(r"Q: (.*?)\nA: (.*?)\n", text)
    return [Document(page_content=f"Q: {q} A: {a}", metadata={"question": q, "answer": a}) for q, a in qa_pairs]

# Set up the model, embeddings, and vector store
def setup_rag_pipeline(qa_documents, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(qa_documents, embedding=embeddings_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Initialize the pipeline with max_new_tokens to control response length
    generator_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=100,  
        device_map="auto"
    )
    
    generator = HuggingFacePipeline(pipeline=generator_pipeline)
    
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=generator, chain_type="stuff", retriever=retriever)

# Main function to run the RAG system
def run_query(qa_chain, user_query):
    query = f"{user_query} Generate just one short phrase. Do not explain anything."
    response = qa_chain.run(query)
    
    answer = response.split("Helpful Answer:")[-1].strip() if "Helpful Answer:" in response else response.strip()

    if not answer or answer == "I don't know":
        answer = "Sorry, I couldn't find an answer to your question."

    return answer


# Streamlit interface
st.title("Diego Rossini's Chatbot")

# Load data and initialize the RAG pipeline only once
qa_documents = load_qa_data("bot/QA.txt") 
qa_chain = setup_rag_pipeline(qa_documents)

# Input field for the user's question
user_query = st.text_input("Enter your question:")

# Process query and display the answer when the user submits a query
if user_query:
    answer = run_query(qa_chain, user_query)
    st.write("Answer:", answer)
