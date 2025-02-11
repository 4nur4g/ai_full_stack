from typing import Annotated

from chromadb import HttpClient
from fastapi import APIRouter, HTTPException, Body
from fastapi import UploadFile, Request
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.utils.llm import get_embedding_model, get_llm

router = APIRouter()

@router.post("/upload-files", summary="Upload one or more text files to the knowledge_base collection")
async def upload_files(
        request: Request,
        file: UploadFile
):
    """
    Accepts one or more text files via form-data (with the field name 'file'), checks whether each document
    (using its filename as a unique identifier in metadata 'source') is already present in the Chroma vector store
    (collection: 'knowledge_base'), and adds it if not.
    """
    db: HttpClient = request.app.state.chroma_client
    collection = db.get_or_create_collection("knowledge_base_v1")
    existing = collection.get(where={"source": file.filename})
    if existing and "ids" in existing and len(existing["ids"]) > 0:
        raise HTTPException(status_code=400, detail="File already added to the collection.")
    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "filename": file.filename,
            "status": "error",
            "message": f"Error reading file: {str(e)}"
        })
    results = []
    document = Document(page_content=content, metadata={"source": file.filename})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents([document])

    Chroma.from_documents(
        client=db,
        collection_name="knowledge_base_v1",
        embedding=get_embedding_model(),
        documents=chunks,
    )
    results.append({
        "filename": file.filename,
        "status": "added",
        "chunks_added": len(chunks),
        "message": "Document added successfully to the knowledge_base collection."
    })

    return {"data": results, "error": False}


class Chat(BaseModel):
    message: str


@router.post("/chat")
async def chat(body: Annotated[Chat, Body()], request: Request):
    """
    Accepts a message in the request body and returns a response from the LLM.
    """
    # Retrieve the Chroma client from the app state.
    db: HttpClient = request.app.state.chroma_client

    vector_store = Chroma(
        client=db,
        collection_name="knowledge_base_v1",
        embedding_function=get_embedding_model(),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    relevant_docs: list[Document] = retriever.invoke(body.message)
    combined_input = (
            "Here are some documents that might help answer the question: "
            + body.message
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([relevant_doc.page_content for relevant_doc in relevant_docs])
            + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )

    # Create a ChatOpenAI model
    model = get_llm()

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)
    return {"data": result.content, "error": False}
