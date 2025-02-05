import os
import pinecone
import openai
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone as PineconeStore

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Load environment variables
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENV = "your_pinecone_environment"
INDEX_NAME = "your_index_name"
OPENAI_API_KEY = "your_openai_api_key"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# Initialize Langchain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Function to load and process data
def load_and_store_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(data)
    
    vector_store = PineconeStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    return "Data ingested successfully"

# API endpoint to ingest data
class IngestData(Resource):
    def post(self):
        url = request.json.get("url")
        if not url:
            return {"error": "URL is required"}, 400
        
        message = load_and_store_data(url)
        return {"message": message}

# API endpoint to chat with the bot
class ChatBot(Resource):
    def post(self):
        query = request.json.get("query")
        if not query:
            return {"error": "Query is required"}, 400
        
        vector_store = PineconeStore(index_name=INDEX_NAME, embedding_function=embeddings.embed_query)
        results = vector_store.similarity_search(query, k=3)
        
        response = [doc.page_content for doc in results]
        return {"response": response}

# Add API routes
api.add_resource(IngestData, "/ingest")
api.add_resource(ChatBot, "/chat")

if __name__ == "__main__":
    app.run(debug=True)
