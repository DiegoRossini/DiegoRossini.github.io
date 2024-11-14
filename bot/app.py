import streamlit as st
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
import re

# Load and parse Q&A data from a text file
@st.cache_data
def load_qa_data(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    qa_pairs = re.findall(r"Q: (.*?)\nA: (.*?)\n", text)
    return [Document(page_content=f"Q: {q} A: {a}", metadata={"question": q, "answer": a}) for q, a in qa_pairs]

# Set up the model, embeddings, and vector store
@st.cache_resource
def setup_rag_pipeline(_qa_documents, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(_qa_documents, embedding=embeddings_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

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

# Main function to handle user queries and generate responses
def run_query(qa_chain, user_query):
    query = f"{user_query} Generate just one short phrase. Do not explain anything."
    response = qa_chain.run(query)

    # Extract the answer from the response
    answer = response.split("Helpful Answer:")[-1].strip() if "Helpful Answer:" in response else response.strip()

    # Remove everything after "You are an AI assistant..." if it exists (case insensitive and flexible)
    unwanted_phrase = "You are an AI assistant"
    if unwanted_phrase in answer:
        answer = answer.split(unwanted_phrase)[0].strip()  # Truncate at the unwanted phrase
    
    return answer if answer else "Sorry, I couldn't find an answer to your question."

# Streamlit interface
st.title("Diego Rossini's personal and sometimes not very clever Chatbot")

# Load data and initialize the RAG pipeline only once
qa_documents = load_qa_data("bot/QA.txt")
qa_chain = setup_rag_pipeline(qa_documents)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Check if a new question has been submitted
if "new_question" not in st.session_state:
    st.session_state["new_question"] = ""

# Input field for the user's question
user_query = st.text_input("Enter your question:", key="user_input", value=st.session_state["new_question"])

# Process the query and update chat history
if user_query:
    # Run query and get the answer
    answer = run_query(qa_chain, user_query)
    
    # Append question and answer to chat history
    st.session_state.chat_history.append({"user": user_query, "bot": answer})
    
    # Clear the input by setting "new_question" to an empty string for the next run
    st.session_state["new_question"] = ""

# Display chat history
st.subheader("Chat History")
for entry in st.session_state.chat_history:
    st.write(f"**User**: {entry['user']}")
    st.write(f"**Bot**: {entry['bot']}")
    st.write("---")
