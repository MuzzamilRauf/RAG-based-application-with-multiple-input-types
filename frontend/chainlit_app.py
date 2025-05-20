import chainlit as cl
import aiohttp
from typing import Optional
import asyncio
from datetime import datetime

API_BASE_URL = "http://127.0.0.1:8080"

@cl.on_chat_start
async def start():
    cl.user_session.set("uploaded_pdf", None)
    files = await cl.AskFileMessage(
        content="Welcome to the RAG Chatbot! Please upload a PDF file to get started.",
        accept=["application/pdf"],
        max_files=1,
        max_size_mb=50,
        timeout=300
    ).send()

    if files:
        file = files[0]
        file_path = file.path
        file_name = file.name

        processing_msg = await cl.Message(content="Processing...").send()
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("file", open(file_path, "rb"), filename=file_name, content_type="application/pdf")
            async with session.post(f"{API_BASE_URL}/upload_pdf", data=form_data) as response:
                await processing_msg.remove()
                if response.status == 200:
                    result = await response.json()
                    cl.user_session.set("uploaded_pdf", file_name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    word_count = len(result["message"].split())
                    await cl.Message(content=f"{result['message']} [{timestamp}] (Words: {word_count})").send()
                else:
                    error = await response.json()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    word_count = len(error["detail"].split())
                    await cl.Message(content=f"Error: {error['detail']} [{timestamp}] (Words: {word_count})").send()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await cl.Message(
            content=f"No PDF uploaded. You can still send text or other files. [{timestamp}] (Words: 11)",
            actions=[cl.Action(name="upload_new_pdf", value="upload", label="Upload New PDF", payload={})]
        ).send()

@cl.action_callback("upload_new_pdf")
async def on_upload_new_pdf(action: cl.Action):
    files = await cl.AskFileMessage(
        content="Please upload a new PDF file to replace the current one.",
        accept=["application/pdf"],
        max_files=1,
        max_size_mb=50,
        timeout=300
    ).send()

    if files:
        file = files[0]
        file_path = file.path
        file_name = file.name

        processing_msg = await cl.Message(content="Processing...").send()
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("file", open(file_path, "rb"), filename=file_name, content_type="application/pdf")
            async with session.post(f"{API_BASE_URL}/upload_pdf", data=form_data) as response:
                await processing_msg.remove()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if response.status == 200:
                    result = await response.json()
                    cl.user_session.set("uploaded_pdf", file_name)
                    word_count = len(result["message"].split())
                    await cl.Message(content=f"{result['message']} [{timestamp}] (Words: {word_count})").send()
                else:
                    error = await response.json()
                    word_count = len(error["detail"].split())
                    await cl.Message(content=f"Error: {error['detail']} [{timestamp}] (Words: {word_count})").send()

@cl.on_message
async def main(message: cl.Message):
    input_type = "text"
    text_data: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None

    if message.elements:
        file = message.elements[0]
        file_path = file.path
        file_name = file.name

        if file_name.endswith(".pdf"):
            processing_msg = await cl.Message(content="Processing...").send()
            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()
                form_data.add_field("file", open(file_path, "rb"), filename=file_name, content_type="application/pdf")
                async with session.post(f"{API_BASE_URL}/upload_pdf", data=form_data) as response:
                    await processing_msg.remove()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if response.status == 200:
                        result = await response.json()
                        cl.user_session.set("uploaded_pdf", file_name)
                        word_count = len(result["message"].split())
                        await cl.Message(content=f"{result['message']} [{timestamp}] (Words: {word_count})").send()
                    else:
                        error = await response.json()
                        word_count = len(error["detail"].split())
                        await cl.Message(content=f"Error: {error['detail']} [{timestamp}] (Words: {word_count})").send()
            return

        elif file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            input_type = "image"
        elif file_name.lower().endswith((".wav", ".mp3")):
            input_type = "voice"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await cl.Message(content=f"Unsupported file type. Use PDFs, images (.jpg, .jpeg, .png), or voice (.wav, .mp3). [{timestamp}] (Words: 15)").send()
            return
    else:
        text_data = message.content.strip()
        if not text_data:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await cl.Message(
                content=f"Please provide some text or upload a file. [{timestamp}] (Words: 10)",
                actions=[cl.Action(name="upload_new_pdf", value="upload", label="Upload New PDF")]
            ).send()
            return

    # Show processing message and keep it active
    processing_msg = cl.Message(content="Processing ...")
    await processing_msg.send()

    # Make async HTTP request with increased timeout and logging
    try:
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("input_type", input_type)
            if input_type == "text":
                form_data.add_field("text_data", text_data)
            elif file_path:
                form_data.add_field("file", open(file_path, "rb"), filename=file_name)

            print(f"DEBUG: Sending request to {API_BASE_URL}/process_input with input_type={input_type}, file={file_name if file_path else None}")
            # Increase timeout to 600 seconds (10 minutes) for voice processing
            async with session.post(
                f"{API_BASE_URL}/process_input",
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                result = await response.json()
                response_content = result.get("response", "No response received") if response.status == 200 else f"Error: {result.get('detail', 'Unknown error')}"
                print(f"DEBUG: Chainlit received response: {response_content}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                word_count = len(response_content.split())
                # Update the message content and call update() without arguments
                processing_msg.content = f"{response_content} [{timestamp}] (Words: {word_count})"
                await processing_msg.update()
    except asyncio.TimeoutError as e:
        error_msg = "Processing took too long (over 10 minutes). Try a shorter audio file or check the server."
        print(f"ERROR: Timeout in Chainlit: {error_msg}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        word_count = len(error_msg.split())
        processing_msg.content = f"{error_msg} [{timestamp}] (Words: {word_count})"
        await processing_msg.update()
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        print(f"ERROR: Chainlit failed to get response: {error_msg}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        word_count = len(error_msg.split())
        processing_msg.content = f"{error_msg} [{timestamp}] (Words: {word_count})"
        await processing_msg.update()

if __name__ == "__main__":
    import os
    os.system("chainlit run chainlit_app.py -h")