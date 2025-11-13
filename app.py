import os
import google.generativeai as genai
import chromadb
import gradio as gr
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash" 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_policies"

# --- Global Initialization ---
try:
    # Initialize generative models
    llm = genai.GenerativeModel(LLM_MODEL)
    
    # Initialize ChromaDB client and get collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("Successfully connected to ChromaDB collection.")
    print(f"Total documents in collection: {collection.count()}")

except Exception as e:
    print(f"Error during initialization: {e}")
    print("Please make sure you have run 'python ingest.py' first.")
    exit()

# --- Core RAG Function ---
def get_rag_response(message, history):
    """
    Handles the RAG logic for a given user query.
    """
    
    # 1. RETRIEVE: Embed the user's query
    try:
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=message,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    except Exception as e:
        print(f"Error embedding query: {e}")
        return "Sorry, I had trouble understanding your question. Please try again."

    # 2. RETRIEVE: Query ChromaDB for relevant documents
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # Retrieve top 3 most relevant chunks
        )
        
        context_chunks = results['documents'][0]
        context_sources = [meta['source'] for meta in results['metadatas'][0]]
        
        if not context_chunks:
            return "Sorry, I couldn't find any relevant information in the company policies."
            
        # Join the retrieved chunks to form the context
        context = "\n---\n".join(context_chunks)
        
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return "Sorry, I had trouble retrieving policy information. Please try again."

    # 3. AUGMENT & GENERATE: Build the prompt and call the LLM
    prompt = f"""
    You are a helpful company policy assistant.
    Your task is to answer the user's question based *only* on the provided context.
    Do not use any external knowledge.
    If the answer is not found in the context, state that you cannot find the information in the policies.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {message}
    
    ANSWER:
    """
    
    try:
        # Generate the content
        response = llm.generate_content(prompt)
        answer = response.text
        
        # 4. CITE SOURCES
        # Get unique source filenames and format them
        unique_sources = ", ".join(sorted(list(set(context_sources))))
        
        # Append sources to the answer
        return f"{answer}\n\n**Sources:** {unique_sources}"
        
    except Exception as e:
        print(f"Error generating response from LLM: {e}")
        return "Sorry, I encountered an error while generating a response."

# --- Gradio Interface ---
# Using gr.ChatInterface adds conversation history automatically
iface = gr.ChatInterface(
    fn=get_rag_response,
    title="Company Policy Chatbot",
    description="Ask me questions about HR, IT, or Remote Work policies.",
    examples=[
        "What is the company's leave policy?",
        "How can employees request remote work?",
        "What is the password policy?"
    ],
    theme="soft"
)

# --- Run the App (Gradio Standalone) ---
if __name__ == "__main__":
    print("Starting Gradio application...")
    # Launch Gradio directly
    iface.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False
    )