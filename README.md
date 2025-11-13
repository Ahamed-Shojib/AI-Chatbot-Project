# RAG Policy Chatbot (Gemini + Chroma + Gradio)

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers employee questions based only on predefined company policy documents.  
It uses **Gemini** for both embedding and generation, **Chroma** as a local vector database, and **Gradio** for a user-friendly web interface.

Unlike frameworks such as LangChain, this implementation is built **manually from scratch**, giving full control over chunking, embedding, vector search, and prompt engineering.

---

## Core Tech Stack

| Component | Tool / Model | Purpose |
| :--- | :--- | :--- |
| **LLM** | `gemini-2.5-flash` | Generates accurate answers using retrieved context. |
| **Embeddings** | `models/text-embedding-004` | Converts text into numerical vectors for similarity search. |
| **Vector Store** | **ChromaDB** | Stores and retrieves policy vectors locally. |
| **Interface** | **Gradio** | Provides a clean and interactive chat UI. |
| **Framework** | **Flask(Python)** | Backend runtime environment. |

---

## Setup & Installation

Follow the steps below to set up and run the project locally.

### Pre-requisites

- **Python 3.8+**
- **Gemini API Key** Obtain one from [Google AI Studio](https://aistudio.google.com/)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ahamed-Shojib/AI-Chatbot-Project.git
cd AI-Chatbot-Project
```

---

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# Windows:
.\env\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

---

### Step 3: Install Dependencies

Install all required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Configuration

You will need to configure your **Gemini API key** for the application to connect to Google generative AI service.

### Create the `.env` File

1. Create a file named `.env` in the project root directory.
2. Add your API key inside it:

```bash
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

---

## Data Setup (One-Time Ingestion)

Before running the chatbot, you must process and store your company policy documents in ChromaDB.

### Ensure the Following Files Exist

- `policy_hr.txt`
- `policy_remote.txt`
- `policy_it.txt`

### Run the Ingestion Script

```bash
python ingest.py
```

This will:
- Load and chunk the policy files  
- Embed them using Gemini embedding model  
- Store the resulting vectors in a `chroma_db/` directory

Once complete, you will see output confirming connection to ChromaDB and total documents added.

---

## Running the Chatbot

After successful ingestion, you can start the chatbot application.

### Run the Main App

```bash
python app.py
```

Once started, Gradio will launch a local web server.

### Access the Interface

Open your browser and visit:

[http://127.0.0.1:7860](http://127.0.0.1:7860)

Now you can chat with your policy bot using queries such as:

- How can employees request remote work?
- What is the company's sick leave policy?
- What is the password complexity requirement?

---

## Project Structure

| File / Folder | Description |
| :--- | :--- |
| **`app.py`** | Main application logic initializes Gemini, connects to ChromaDB, performs RAG, and runs the Gradio interface. |
| **`ingest.py`** | Preprocessing script reads, chunks, embeds policy documents, and populates the Chroma vector store. |
| **`requirements.txt`** | Contains all necessary dependencies (`google-generativeai`, `chromadb`, `gradio`, `python-dotenv`). |
| **`.env`** | Stores your Gemini API key (never share or commit this file). |
| **`chroma_db/`** | The persistent vector database storing embeddings locally. *(Do not delete unless re-ingesting data.)* |

---

## Example Usage

**User:** What is the company work-from-home policy? 
**Chatbot:** According to the Remote Work Policy, employees may request up to 3 days per week of remote work with manager approval.

---

## Troubleshooting

- **`GOOGLE_API_KEY not found`** Ensure `.env` exists and the key is correctly defined.  
- **Empty or missing Chroma database** Re-run `python ingest.py`.  
- **Gradio UI not launching** Check your Python environment or confirm port `7860` is not in use.

---

## License

This project is open-source and available for personal or educational use.  
You may modify or extend it for internal company chatbot deployments.

---

## Acknowledgements

- [Google Gemini API](https://aistudio.google.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)

---

Made with by [Mehedi Hasan Shojib](https://www.linkedin.com/in/mehedi-hasan-shojib/)
