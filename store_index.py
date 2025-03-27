import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")  # Update based on your setup

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medicalbot"

# Check if the index exists, create if not
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=768,  # Adjust based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Update as needed
    )
else:
    print(f"Index {index_name} already exists!")

# Connect to the Pinecone index
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# Process PDF and extract text chunks
extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone Vector Store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

print("Pinecone setup complete and ready for use.")
