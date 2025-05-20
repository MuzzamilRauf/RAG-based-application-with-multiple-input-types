import chainlit as cl
import aiohttp
from typing import Optional
import asyncio
from datetime import datetime

API_BASE_URL = "http://127.0.0.1:8080"

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def word_count(text: str) -> int:
    return len(text.split())

async def send_processing_msg():
    return await cl.Message(content="Processing...").send()

async def process_file_upload(file_path, file_name, endpoint):
    async with aiohttp.ClientSession() as session:
        form_data = aiohttp.FormData()
        form_data.add_field("file", open(file_path, "rb"), filename=file_name, content_type="application/pdf")
        async with session.post(f"{API_BASE_URL}/{endpoint}", data=form_data) as response:
            return response

async def handle_response(response, success_msg_prefix="Success"):
    timestamp = get_timestamp()
    result = await response.json()
    content = result.get("message" if response.status == 200 else "detail", "Unknown response")
    return f"{content} [{timestamp}] (Words: {word_count(content)})"

# -------------------- CHAT START --------------------
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
        processing_msg = await send_processing_msg()
        response = await process_file_upload(file.path, file.name, "upload_pdf")
        await processing_msg.remove()
        final_msg = await handle_response(response)
        if response.status == 200:
            cl.user_session.set("uploaded_pdf", file.name)
        await cl.Message(content=final_msg).send()
    else:
        timestamp = get_timestamp()
        await cl.Message(
            content=f"No PDF uploaded. You can still send text or other files. [{timestamp}] (Words: 11)",
            actions=[cl.Action(name="upload_new_pdf", value="upload", label="Upload New PDF", payload={})]
        ).send()

# -------------------- ACTION: UPLOAD NEW PDF --------------------
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
        processing_msg = await send_processing_msg()
        response = await process_file_upload(file.path, file.name, "upload_pdf")
        await processing_msg.remove()
        final_msg = await handle_response(response)
        if response.status == 200:
            cl.user_session.set("uploaded_pdf", file.name)
        await cl.Message(content=final_msg).send()

# -------------------- MAIN INPUT HANDLER --------------------
@cl.on_message
async def main(message: cl.Message):
    input_type = "text"
    text_data = None
    file_path = None
    file_name = None

    # -------------------- FILE HANDLING --------------------
    if message.elements:
        file = message.elements[0]
        file_path = file.path
        file_name = file.name.lower()

        if file_name.endswith(".pdf"):
            processing_msg = await send_processing_msg()
            response = await process_file_upload(file_path, file_name, "upload_pdf")
            await processing_msg.remove()
            final_msg = await handle_response(response)
            if response.status == 200:
                cl.user_session.set("uploaded_pdf", file_name)
            await cl.Message(content=final_msg).send()
            return
        elif file_name.endswith((".jpg", ".jpeg", ".png")):
            input_type = "image"
        elif file_name.endswith((".wav", ".mp3")):
            input_type = "voice"
        else:
            timestamp = get_timestamp()
            await cl.Message(content=f"Unsupported file type. Use PDFs, images (.jpg, .jpeg, .png), or voice (.wav, .mp3). [{timestamp}] (Words: 15)").send()
            return
    else:
        text_data = message.content.strip()
        if not text_data:
            timestamp = get_timestamp()
            await cl.Message(
                content=f"Please provide some text or upload a file. [{timestamp}] (Words: 10)",
                actions=[cl.Action(name="upload_new_pdf", value="upload", label="Upload New PDF")]
            ).send()
            return

    # -------------------- PROCESSING MULTI-MODAL INPUT --------------------
    processing_msg = cl.Message(content="Processing ...")
    await processing_msg.send()

    try:
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("input_type", input_type)
            if input_type == "text":
                form_data.add_field("text_data", text_data)
            elif file_path:
                form_data.add_field("file", open(file_path, "rb"), filename=file_name)

            print(f"DEBUG: Sending request to {API_BASE_URL}/process_input with input_type={input_type}")
            async with session.post(
                f"{API_BASE_URL}/process_input",
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                result = await response.json()
                response_content = (
                    result.get("response", "No response received") if response.status == 200
                    else f"Error: {result.get('detail', 'Unknown error')}"
                )
                print(f"DEBUG: Chainlit received response: {response_content}")
                timestamp = get_timestamp()
                wc = word_count(response_content)
                processing_msg.content = f"{response_content} [{timestamp}] (Words: {wc})"
                await processing_msg.update()
    except asyncio.TimeoutError:
        error_msg = "Processing took too long (over 10 minutes). Try a shorter audio file or check the server."
        timestamp = get_timestamp()
        wc = word_count(error_msg)
        processing_msg.content = f"{error_msg} [{timestamp}] (Words: {wc})"
        await processing_msg.update()
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        timestamp = get_timestamp()
        wc = word_count(error_msg)
        processing_msg.content = f"{error_msg} [{timestamp}] (Words: {wc})"
        await processing_msg.update()

# -------------------- RUN SCRIPT LOCALLY --------------------
if __name__ == "__main__":
    import os
    os.system("chainlit run chainlit_app.py -h")
