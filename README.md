# InQuireAI
InQuireAI autonomously processes uploaded documents, analyzes their content, and generates high-level interview questions and answers based solely on that context. This goes beyond a simple chatbot or assistant because it involves a chain-of-thought process—from document ingestion and vectorization to retrieval and dynamic content generation

## Features

- **Document Upload:** Easily upload a PDF user-friendly web interface.
- **Automated Analysis:** Uses LangChain with a HuggingFace opensourced language model (Mistral-7B-Instruct-v0.3) to analyze the document content.
- **Dynamic Q&A Generation:** Produces exactly diverse, insightful question-answer pairs directly derived from the document.
- **Persistent Vector Store with Explicit Updates:** Utilizes ChromaDB as a persistent vector store, which is reinitialized for every new upload to ensure fresh and relevant outputs.


- **Backend:** FastAPI, Uvicorn
- **AI & NLP:** LangChain, HuggingFace Endpoint (Mistral-7B-Instruct-v0.3)
- **Vector DB:** FAISS/Chroma for document embeddings and retrieval



## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/InQuireAI.git
   cd InQuireAI

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

pip install -r requirements.txt

Configure Environment Variables:

Create a .env file in the root directory and add your HuggingFace API token:

HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"


Run the Application:

uvicorn app:app --reload
Access the Web Interface:

Open your browser and navigate to http://localhost:8080

How It Works
