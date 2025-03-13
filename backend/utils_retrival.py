# # import pinecone
# # import asyncio
# # from sqlalchemy import create_engine
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import Pinecone
# # from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
# # from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, HF_EMBEDDING_MODEL, HF_API_KEY, \
# #     DB_CONNECTION_STRING
# #
# # # Initialize Pinecone
# # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# # index = pinecone.Index(PINECONE_INDEX_NAME)
# #
# # # Load Hugging Face Embedding Model
# # embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
# #
# #
# # # Text Splitter
# # def split_text(docs, chunk_size=500, chunk_overlap=50):
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# #     return splitter.split_documents(docs)
# #
# #
# # # 1️⃣ Load PDF using LangChain
# # async def load_pdf(file_path):
# #     loader = PyPDFLoader(file_path)
# #     docs = []
# #     async for page in loader.alazy_load():
# #         docs.append(page)
# #     return docs
# #
# #
# # # 2️⃣ Load Web URL using LangChain
# # async def load_web_url(url):
# #     loader = WebBaseLoader(web_paths=[url])
# #     docs = []
# #     async for doc in loader.alazy_load():
# #         docs.append(doc)
# #     return docs
# #
# #
# # # 3️⃣ Load CSV using LangChain
# # def load_csv(file_path):
# #     loader = CSVLoader(file_path=file_path)
# #     return loader.load()
# #
# #
# # # 4️⃣ Load Database
# # def load_database(query):
# #     engine = create_engine(DB_CONNECTION_STRING)
# #     with engine.connect() as conn:
# #         result = conn.execute(query)
# #         rows = result.fetchall()
# #         return " ".join([" ".join(map(str, row)) for row in rows])
# #
# #
# # # Generate Embeddings and Store in Pinecone
# # def generate_and_store_embeddings(docs):
# #     chunks = split_text(docs)
# #     vector_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])
# #
# #     for i, chunk in enumerate(chunks):
# #         index.upsert([(f"doc_{i}", vector_embeddings[i], {"text": chunk.page_content})])
# #
# #     return f"Stored {len(chunks)} text chunks in Pinecone."
# #
# #
# # # Wrapper function to process and store any data type
# # async def process_and_store(data_type, source):
# #     if data_type == "pdf":
# #         docs = await load_pdf(source)
# #     elif data_type == "web":
# #         docs = await load_web_url(source)
# #     elif data_type == "csv":
# #         docs = load_csv(source)
# #     elif data_type == "database":
# #         text = load_database(source)
# #         docs = [{"page_content": text}]
# #     else:
# #         raise ValueError("Unsupported data type")
# #
# #     return generate_and_store_embeddings(docs)
#
#
# import os
# # from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import Pinecone
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
#
# # Function 1: Load PDF
# def load_pdf(file_path):
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
#     return documents
#
#
# # Function 2: Split Text into Chunks
# def split_text(documents, chunk_size=500, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_documents(documents)
#
#
# # Function 3: Generate Embeddings using Hugging Face
# def generate_embeddings(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#     return embedding_model, [embedding_model.embed_query(doc.page_content) for doc in docs]
#
#
# # Function 4: Initialize Pinecone and Create Index
# def initialize_pinecone(api_key, index_name, cloud="aws", region="us-east-1"):
#     pc = Pinecone(api_key=api_key)
#
#     if not pc.has_index(index_name):
#         print(f"Creating Pinecone index: {index_name}")
#         pc.create_index_for_model(
#             name=index_name,
#             cloud=cloud,
#             region=region,
#             embed={
#                 "model": "llama-text-embed-v2",
#                 "field_map": {"text": "chunk_text"}
#             }
#         )
#     return pc, index_name
#
#
# # Function 5: Store Embeddings in Pinecone
# def store_embeddings_in_pinecone(pc, index_name, docs, embedding_model):
#     # Extract text content from Document objects
#     texts = [doc.page_content for doc in docs]
#
#     # Generate embeddings
#     doc_embeddings = embedding_model.embed_documents(texts)
#
#     # Prepare data for Pinecone
#     vectors = [
#         (str(i), embedding.tolist(), {"text": text})
#         for i, (embedding, text) in enumerate(zip(doc_embeddings, texts))
#     ]
#
#     # Upsert into Pinecone
#     index = pc.Index(index_name)
#     index.upsert(vectors)
#     print("Embeddings stored in Pinecone successfully!")
#
#
# # Main Execution
# if __name__ == "__main__":
#     PDF_PATH = "The-48-Laws-Of-Power.pdf"
#     PINECONE_API_KEY = "pcsk_52qZAL_JCtn4vYQVXaFzo5JrmFMx7pU7XYY5snbPMfK4diUUtqzbJwsiXkcF9BgcSgqCTS"
#     INDEX_NAME = "dense-index"
#
#     # Load and process PDF
#     documents = load_pdf(PDF_PATH)
#     docs = split_text(documents)
#
#     # Generate embeddings
#     embedding_model, doc_embeddings = generate_embeddings(docs)
#
#     # Initialize Pinecone and create index
#     pc, index_name = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)
#
#     # Store embeddings in Pinecone
#     store_embeddings_in_pinecone(pc, index_name, docs, embedding_model)
#
#

import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["PINECONE_API_KEY"] = "pcsk_4PGG2i_4NSfCD5Q6PsCs8bsjcRpjgsUUSVYsa4m2AGbTuvnPba8g182Fm9jGVHXgAyHtKn"

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "dense-index"
PDF_PATH = "The-48-Laws-Of-Power.pdf"


# Function 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# Function 2: Split Text into Chunks
def split_text(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


# Function 3: Generate Embeddings
def generate_embeddings(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model, embedding_model.embed_documents([doc.page_content for doc in docs])


# Function 4: Initialize Pinecone and Create Index
def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Update this to match your embedding model
            metric="cosine",  # Ensure metric is consistent with embedding model
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as neede
        )

    index = pc.Index(index_name)
    return pc, index


# Function 5: Store Embeddings in Pinecone
def store_embeddings_in_pinecone(index, docs, embedding_model, batch_size=100):
    texts = [doc.page_content for doc in docs]

    doc_embeddings = embedding_model.embed_documents(texts)

    vectors = [
        (str(i), np.array(embedding).tolist(), {"text": text})
        for i, (embedding, text) in enumerate(zip(doc_embeddings, texts))
    ]

    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(batch)
        print(f"Stored batch {i // batch_size + 1} of {len(vectors) // batch_size + 1}")

    print("Embeddings stored in Pinecone successfully!")


# Main Execution
if __name__ == "__main__":
    # Load and process PDF
    documents = load_pdf(PDF_PATH)
    docs = split_text(documents)

    # Generate embeddings
    embedding_model, doc_embeddings = generate_embeddings(docs)

    # Initialize Pinecone and create index
    pc, index = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)

    # Store embeddings in Pinecone
    store_embeddings_in_pinecone(index, docs, embedding_model)
