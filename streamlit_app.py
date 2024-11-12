import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import re

# Load and process Q&A document
@st.cache_data
def load_qa_data(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    qa_pairs = re.findall(r"Q: (.*?)\nA: (.*?)\n", text)
    qa_documents = [Document(page_content=f"Q: {q} A: {a}", metadata={"question": q, "answer": a}) for q, a in qa_pairs]
    return qa_documents

# Initialize the embedding model
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Load the model and tokenizer
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    generator_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024, device_map="auto")
    generator = HuggingFacePipeline(pipeline=generator_pipeline)
    return generator

# Streamlit user interface
def main():
    st.title("Retrieval-Augmented Generation (RAG) Chatbot")

    # File uploader for the Q&A document
    uploaded_file = st.file_uploader("Upload your Q&A text file", type=["txt"])

    if uploaded_file is not None:
        # Load Q&A data from the uploaded file
        qa_documents = load_qa_data(uploaded_file)
        
        # Create Chroma vectorstore
        embeddings_model = get_embedding_model()
        qa_vectorstore = Chroma.from_documents(qa_documents, embedding=embeddings_model)

        # Load the generator model
        generator = load_model()

        # Set up the retriever
        qa_retriever = qa_vectorstore.as_retriever()

        # Set up the RAG pipeline
        qa_chain = RetrievalQA.from_chain_type(
            llm=generator,
            chain_type="stuff",
            retriever=qa_retriever
        )

        # Query input
        user_query = st.text_input("Ask a question:")

        if user_query:
            # Get response from the QA chain
            response = qa_chain.run(user_query)
            answer = response.split("Helpful Answer:")[-1].strip() if "Helpful Answer:" in response else response
            st.write(f"**Answer:** {answer}")

# Run the app
if __name__ == "__main__":
    main()
