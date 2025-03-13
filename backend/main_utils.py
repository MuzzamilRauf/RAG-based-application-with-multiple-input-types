import os
import base64
import whisper
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# 082e293b6aa70fe726d44fcc8e886bd9dfa542be221643c470e320ee75706f17
os.environ["TOGETHER_API_KEY"] = "use your own token"
os.environ["PINECONE_API_KEY"] = "use your own token"
# pcsk_4PGG2i_4NSfCD5Q6PsCs8bsjcRpjgsUUSVYsa4m2AGbTuvnPba8g182Fm9jGVHXgAyHtKn

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            openai_api_base="https://api.together.xyz/v1",
            openai_api_key=os.environ["TOGETHER_API_KEY"],
            temperature=0.7,
            max_tokens=1000  # Increased to ensure room for detailed responses
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.whisper_model = whisper.load_model("small")
        self.pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = "dense-index"
        self.index = self.initialize_pinecone()
        self.memory = ConversationBufferMemory(return_messages=True)

    def process_text(self, text):
        return text

    def process_image_for_text_extraction(self, image_path):
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_input = f"data:image/jpg;base64,{image_data}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract all text from this image accurately and completely, preserving the original wording and structure."},
                {"type": "image_url", "image_url": {"url": image_input}},
            ]
        )
        response = self.llm.invoke([message])
        extracted_text = response.content
        print(f"DEBUG: Extracted text from image '{image_path}': {extracted_text}")
        return extracted_text

    def process_audio(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.whisper_model, mel, options)
        return result.text

    def generate_embedding_of_input(self, text):
        embeddings = self.embedding_model.embed_documents([str(text)])
        return embeddings[0]

    def Query_into_text_and_embeddings(self, input_type, input_data):
        if input_type == "text":
            processed_text = self.process_text(input_data)
        elif input_type == "image":
            processed_text = self.process_image_for_text_extraction(input_data)
        elif input_type == "voice":
            processed_text = self.process_audio(input_data)
        else:
            raise ValueError("Invalid input type. Choose from 'text', 'image', or 'voice'.")
        embeddings = self.generate_embedding_of_input(processed_text)
        return processed_text, embeddings

    def initialize_pinecone(self):
        if self.index_name not in [i["name"] for i in self.pinecone.list_indexes()]:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pinecone.Index(self.index_name)

    def clear_pinecone_index(self):
        try:
            self.index.delete(delete_all=True)
            print("Previous embeddings cleared from Pinecone successfully!")
        except Exception as e:
            print(f"Failed to clear Pinecone index: {str(e)}")

    def store_embeddings_in_pinecone(self, docs):
        texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embedding_model.embed_documents(texts)
        vectors = [(str(i), np.array(embedding).tolist(), {"text": text}) for i, (embedding, text) in
                   enumerate(zip(doc_embeddings, texts))]
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors[i:i + 100])
        print("Embeddings stored in Pinecone successfully!")

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_text(self, documents, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def process_pdf_and_store_embeddings(self, pdf_path):
        self.clear_pinecone_index()
        documents = self.load_pdf(pdf_path)
        docs = self.split_text(documents)
        self.store_embeddings_in_pinecone(docs)

    def validate_embedding(self, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        if len(embedding) != 768:
            raise ValueError(f"Embedding dimension mismatch: {len(embedding)}, expected 768")
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return embedding.tolist()

    def retrieve_and_generate_response(self, input_type, input_data):
        query, query_embedding = self.Query_into_text_and_embeddings(input_type, input_data)
        query_embeddings = self.validate_embedding(query_embedding)

        if len(query_embeddings) != 768:
            raise ValueError(f"Incorrect embedding size: {len(query_embeddings)}, expected 768")

        try:
            results = self.index.query(vector=query_embeddings, top_k=5, include_metadata=True)
        except Exception as e:
            print(f"Pinecone query failed: {str(e)}")
            raise

        # Get previous conversation from memory (limit to last 4 messages)
        conversation_history = self.memory.load_memory_variables({})["history"][-4:]
        history_text = "\n".join(
            [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in conversation_history])

        # Determine if the query is relevant to the PDF context
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        top_score = results["matches"][0]["score"] if results["matches"] else 0.0

        # Enhanced prompt with explicit instructions
        if top_score > 0.7:
            prompt = f"""You are a helpful Retrieval Augmented Generation (RAG) AI assistant and you will get the context from uploaded PDF with user Query so please gives to the point helpful informative answer to the user query and never answer in one word:

            Previous conversation:
            {history_text}

            PDF Context:
            {context}

            Question: {query}
            """
        else:
            prompt = f"""You are a helpful AI assistant so please gives to the point helpful informative answer to the user query and never answer in one word:

            Previous conversation:
            {history_text}

            Question: {query}
            """

        # Generate response
        response = self.llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
        response_content = response.content
        print(f"DEBUG: Generated response for '{query}': {response_content}")

        # Store the query and response in memory
        self.memory.save_context({"input": query}, {"output": response_content})

        return response_content


if __name__ == "__main__":
    rag_obj = RAGPipeline()
    text_input = "Give me the introduction about research paper."
    pdf_input = "PDF_Data/research paper.pdf"
    image_input = "Input_Images/English_PDF_Question.JPG"
    print("Processing Given Input:")
    print(rag_obj.retrieve_and_generate_response("text", text_input))