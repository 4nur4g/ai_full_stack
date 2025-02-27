import os
import tempfile
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Body, File
from fastapi import UploadFile, Request
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from rich.pretty import pprint

from app.agents.rag import get_graph
from app.utils.llm import get_embedding_model, get_llm

router = APIRouter()

collection_name = "knowledge_base_v10"
thread_id = str(45)
chunk_size = 1000
chunk_overlap = 100


@router.post("/upload-files", summary="Upload one or more text files to the knowledge_base collection")
async def upload_files(
        request: Request,
        files: list[UploadFile] = File(...),
):
    """
    Accepts one or more files via form-data (field name 'files'),
    checks if each document (using its filename as a unique identifier in metadata 'source')
    is already in the Chroma vector store (collection: 'knowledge_base_v4'),
    and adds it if not. Files are temporarily saved on disk before processing.
    """
    if not (1 <= len(files) <= 5):
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 files.")

    db = request.app.chroma_client
    vector_store = Chroma(
        client=db,
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = []

    for file in files:
        # Check if file already exists based on its filename.
        existing = vector_store.get(where={"source": file.filename}, limit=1)
        if existing and existing.get("ids"):
            results.append({
                "filename": file.filename,
                "status": "exists",
                "message": "File already added to the collection."
            })
            continue

        try:
            content_bytes = await file.read()
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Error reading file: {str(e)}"
            })
            continue

        # Save file temporarily on disk.
        # The suffix is determined from the original filename to preserve file type.
        suffix = os.path.splitext(file.filename)[1]
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(content_bytes)
                temp_file_path = tmp_file.name
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Error saving temporary file: {str(e)}"
            })
            continue

        # Process file based on its content type.
        if file.content_type == "text/plain":
            try:
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Decoding error: {str(e)}"
                })
                os.remove(temp_file_path)
                continue
            documents = [Document(page_content=content, metadata={"source": file.filename})]
        elif file.content_type == "application/pdf":
            try:
                loader = UnstructuredPDFLoader(temp_file_path)
                documents = loader.load()
                # Ensure each document carries the filename as metadata.
                for document in documents:
                    document.metadata["source"] = file.filename
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Error processing PDF: {str(e)}"
                })
                os.remove(temp_file_path)
                continue
        else:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported file type: {file.content_type}"
            })
            os.remove(temp_file_path)
            continue

        # Remove the temporary file after processing.
        os.remove(temp_file_path)

        # Split the document(s) into chunks and add to the vector store.
        chunks = text_splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        results.append({
            "filename": file.filename,
            "status": "added",
            "chunks_added": len(chunks),
            "message": "Document added successfully to the knowledge_base collection."
        })

    return {"data": results, "error": False}


class Chat(BaseModel):
    message: str
    stream: bool = True


@router.post("/chat")
async def chat(body: Annotated[Chat, Body()], request: Request):
    """
    Accepts a message in the request body and returns a response from the LLM.
    """
    graph: CompiledStateGraph = await get_graph(get_llm(), get_embedding_model(), request.app.chroma_client, collection_name, request)

    config = {"configurable": {"thread_id": thread_id}}
    if not body.stream:
        result = await graph.ainvoke({"messages": [{"role": "user", "content": body.message}]}, config)
        pprint(result)
        return {"data": result["messages"][-1].content, "error": False}

    async def event_generator():
        async for message, metadata in graph.astream(
                {"messages": [{"role": "user", "content": body.message}]},
                stream_mode="messages",
                config=config,
        ):
            if metadata["langgraph_node"] in ["query_or_respond", "generate"]:
                content = message.content
                yield f"data: {content}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
