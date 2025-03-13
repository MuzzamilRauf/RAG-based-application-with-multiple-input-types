import os
import base64
import whisper
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from huggingface_hub import login

# Set your Together.ai API key as environment variable
os.environ["TOGETHER_API_KEY"] = "082e293b6aa70fe726d44fcc8e886bd9dfa542be221643c470e320ee75706f17"

class RAGPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline with Together AI's Qwen model.
        """
        # Load the LLM (Qwen model from Together AI)
        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-VL-72B-Instruct",  # Specify the Qwen2.5-VL-7B model
            openai_api_base="https://api.together.xyz/v1",  # Together.ai API base URL
            openai_api_key=os.environ["TOGETHER_API_KEY"],  # API key from environment
        )

        # Load the embedding model (you can change this as needed)
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        # Load Whisper model for voice-to-text conversion
        self.whisper_model = whisper.load_model("base")

    def process_text(self, text):
        """
        Process raw text input.
        """
        return text

    def process_image_for_text_extraction(self, image_path):
        """
        Extract text from an image using the Qwen model.
        """
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_input = f"data:image/jpg;base64,{image_data}"

        message = HumanMessage(
            content=[
                {"type": "text", "text" : "You are an Optical Character Recognition Assistant so, Extract the text from this image"},
                {"type": "image_url", "image_url": {"url": image_input}},
            ]
        )

        # Invoke the model with the multimodal input
        response = self.llm.invoke([message])
        return response.content

    def process_audio(self, audio_path):
        """
        Convert speech to text using Whisper.
        """

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)  # Ensure the audio is properly sized

        # Convert to mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)

        # Decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.whisper_model, mel, options)

        return result.text

        # result = self.whisper_model.transcribe(audio_path)
        # return result["text"]

    def generate_embedding(self, text):
        """
        Generate embeddings for the extracted text.
        """
        # return self.embedding_model.embed_documents([Document(page_content=text)])

        embeddings = self.embedding_model.embed_documents([text])
        return embeddings[0]


    def Query_into_text_and_embeddings(self, input_type, input_data):
        """
        Process different input types and generate embeddings.
        """
        if input_type == "text":
            processed_text = self.process_text(input_data)
        elif input_type == "image":
            processed_text = self.process_image_for_text_extraction(input_data)
        elif input_type == "voice":
            processed_text = self.process_audio(input_data)
        else:
            raise ValueError("Invalid input type. Choose from 'text', 'image', or 'voice'.")

        # Generate embeddings
        embeddings = self.generate_embedding(processed_text)
        return processed_text, embeddings


# Example Usage
if __name__ == "__main__":
    # TOGETHER_API_KEY = "your_together_ai_api_key"  # Replace with your actual API key

    login("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")

    rag = RAGPipeline()

    # Example inputs
    text_input = "This is a test input. My name is Muzzamil Rauf. I live in Pakistan. Currently I am living in Pakistan."
    input_image = "backend/Fotos De Zianya En _3  97.jpg"
    audio_input = "backend/harvard.wav"

    print("Processing Text Input:")
    text_result, text_embeddings = rag.Query_into_text_and_embeddings("text", text_input)
    print("Extracted Text:", text_result)
    print("Text Embeddings:", text_embeddings)

    print("\nProcessing Image Input:")
    image_result, image_embeddings = rag.Query_into_text_and_embeddings("image", input_image)
    print("Extracted Text from Image:", image_result)

    print("\nProcessing Voice Input:")
    audio_result, audio_embeddings = rag.Query_into_text_and_embeddings("voice", audio_input)
    print("Extracted Text from Audio:", audio_result)
