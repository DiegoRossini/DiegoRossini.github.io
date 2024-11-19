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
def setup_rag_pipeline(_qa_documents, model_name="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"):
    # Initialize the embedding model and vector store
    embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(_qa_documents, embedding=embeddings_model)
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    
    # Define a custom generator using the HuggingFacePipeline
    def custom_generator(user_query):
        # Prepare the messages in the expected format
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ]
        
        # Apply the chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and generate response
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=30,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    # Return the retriever and custom generator function
    retriever = vectorstore.as_retriever()
    return retriever, custom_generator

# Main function to handle user queries and generate responses
def run_query(retriever, generator, user_query):
    # Retrieve context using the retriever
    docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Combine user query with retrieved context
    query_with_context = f"Based on the following context :\n{context}\n\n{user_query}.\nGenerate just one short phrase. Do not explain anything."
    
    # Generate the response using the custom generator
    response = generator(query_with_context)
    return response if response else "Sorry, I couldn't find an answer to your question."

# Streamlit interface
st.title("Diego Rossini's personal Chatbot")

# Load data and initialize the RAG pipeline only once
qa_documents = load_qa_data("bot/QA.txt")
retriever, custom_generator = setup_rag_pipeline(qa_documents)

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
    answer = run_query(retriever, custom_generator, user_query)
    
    # Append question and answer to chat history
    st.session_state.chat_history.append({"user": user_query, "bot": answer})
    
    # Clear the input by setting "new_question" to an empty string for the next run
    st.session_state["new_question"] = ""

# Display chat history
st.subheader("Chat History")
for entry in reversed(st.session_state.chat_history):
    st.write(f"**User**: {entry['user']}")
    st.write(f"**Bot**: {entry['bot']}")
    st.write("---")