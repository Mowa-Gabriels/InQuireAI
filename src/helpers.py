from langchain.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint # Now the import should work
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain import embeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain.vectorstores import FAISS

import os,shutil
import dotenv
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


import os

# Get the API key from the environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Check if the API key was found
if HUGGINGFACEHUB_API_TOKEN:
    print("API key found.")
    # Now you can use OPENAI_API_KEY in your code
else:
    print("API key not found in .env file.")
    HUGGINGFACEHUB_API_TOKEN = input("Please enter your HuggingFcae API token: ")
    # You might want to store the entered key in the .env file for future use
    # but be careful about security implications if you're sharing the file.
    with open('.env', 'a') as f:
        f.write(f'\nHUGGINGFACEHUB_API_TOKEN="{HUGGINGFACEHUB_API_TOKEN}"')
    print("API key stored in .env file for future use.")


def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                                   chunk_overlap=20)
    
    text_chunk=text_splitter.split_documents(data)

    return text_chunk

def llm_pipeline(file_path):

    text_chunk = file_processing(file_path)

    huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceEndpoint(

    task='text-generation',
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=1024,
    temperature=0.3,
    huggingfacehub_api_token=huggingfacehub_api_token
    )
    
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(text_chunk, emb_model)
    
    retriever = vector_store.as_retriever()

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context"]
    )

    # Update the RetrievalQA chain with the new prompt
    query_retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT} 
    )

    # Run the query to generate questions and answers
    result = query_retriever_chain({"query": "Generate 10 high-level Q&A pairs based on the context."}) 
    qa_list = result["result"].split("\n\n")# Split into question-answer pairs

    return result, qa_list
        

        