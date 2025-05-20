import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from main_utils import RAGPipeline  # Correct import (assuming utils.py)

app = FastAPI()
rag_pipeline = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        print("RAGPipeline initialized successfully on startup!")
    except Exception as e:
        print(f"Failed to initialize RAGPipeline: {str(e)}")
        raise


@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "API is running"}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        rag_pipeline.process_pdf_and_store_embeddings(pdf_path)
        return {"message": f"PDF {file.filename} uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/process_input")
async def process_input(
        input_type: str = Form(...),
        text_data: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None)
):
    try:
        valid_input_types = ["text", "image", "voice"]
        if input_type not in valid_input_types:
            raise HTTPException(status_code=400, detail="Invalid input_type. Must be 'text', 'image', or 'voice'")

        if input_type == "text":
            if not text_data:
                raise HTTPException(status_code=400, detail="text_data is required for input_type 'text'")
            input_data = text_data
        elif input_type in ["image", "voice"]:
            if not file:
                raise HTTPException(status_code=400, detail="File is required for input_type 'image' or 'voice'")
            if input_type == "image" and not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, or .png files allowed for images")
            if input_type == "voice" and not file.filename.lower().endswith(('.wav', '.mp3')):
                raise HTTPException(status_code=400, detail="Only .wav or .mp3 files allowed for voice")
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_data = file_path
        else:
            raise HTTPException(status_code=400, detail="Invalid input_type")

        response = rag_pipeline.retrieve_and_generate_response(input_type, input_data)
        print(f"DEBUG: Response for {input_type} input: {response}")

        if input_type in ["image", "voice"] and os.path.exists(file_path):
            os.remove(file_path)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)