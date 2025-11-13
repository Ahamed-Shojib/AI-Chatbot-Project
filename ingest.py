import os
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Configuration ---
POLICY_FILES = ["policy_hr.txt", "policy_remote.txt", "policy_it.txt"]
EMBEDDING_MODEL = "models/text-embedding-004"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_policies"

def load_and_chunk_files(file_list):
    """
    Loads text files and splits them into chunks based on double newlines.
    Each chunk is stored with metadata indicating its source file.
    """
    documents = []
    metadatas = []
    ids = []
    
    print("Loading and chunking documents...")
    for filename in file_list:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Simple chunking strategy: split by double newline (paragraphs/sections)
            chunks = text.split('\n\n')
            
            for i, chunk in enumerate(chunks):
                # Filter out empty or very short chunks
                if chunk.strip():
                    doc_id = f"{filename}-chunk-{i}"
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(doc_id)
                    
            print(f"Loaded and chunked {filename}")
            
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return documents, metadatas, ids

def get_gemini_embeddings(texts, model_name):
    """
    Gets embeddings for a list of texts using the Gemini API.
    We specify task_type="RETRIEVAL_DOCUMENT" for documents we want to store.
    """
    print(f"Generating embeddings for {len(texts)} documents...")
    try:
        # Note: The 'title' parameter is optional but can improve embeddings
        result = genai.embed_content(
            model=model_name,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def main():
    """
    Main function to orchestrate the ingestion pipeline.
    """
    print("Starting ingestion process...")
    
    # 1. Load and Chunk
    documents, metadatas, ids = load_and_chunk_files(POLICY_FILES)
    
    if not documents:
        print("No documents were loaded. Exiting.")
        return

    # 2. Get Embeddings
    embeddings = get_gemini_embeddings(documents, EMBEDDING_MODEL)
    
    if embeddings is None:
        print("Failed to generate embeddings. Exiting.")
        return

    # 3. Store in Chroma
    print(f"Storing {len(documents)} chunks in ChromaDB...")
    try:
        # Create a persistent client. Data will be saved to disk
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Get or create the collection
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
        # Add the documents to the collection
        # Note: ChromaDB handles batching internally if the list is very large.
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print("\n--- Ingestion Complete ---")
        print(f"Total chunks added: {collection.count()}")
        print(f"Collection '{COLLECTION_NAME}' is ready at {CHROMA_PATH}")
        
    except Exception as e:
        print(f"Error storing data in ChromaDB: {e}")

if __name__ == "__main__":
    main()