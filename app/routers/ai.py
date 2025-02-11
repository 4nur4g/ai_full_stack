import uuid

from chromadb import AsyncClientAPI
from chromadb.api.models import AsyncCollection
from fastapi import APIRouter, HTTPException
from fastapi import UploadFile, Request
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from app.utils.chroma import LangChainEmbeddingAdapter

router = APIRouter()

EMBEDDING_MODEL = "nomic-embed-text"


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
    # Retrieve the Chroma client from the app state.
    db: AsyncClientAPI = request.app.state.chroma_client

    collection: AsyncCollection = await db.get_or_create_collection("test",
                                                                    embedding_function=LangChainEmbeddingAdapter(
                                                                        OllamaEmbeddings(model=EMBEDDING_MODEL)))
    results = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    filename = file.filename

    print(filename)
    content = None
    # Read file content (assuming UTF-8 encoding).
    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "filename": filename,
            "status": "error",
            "message": f"Error reading file: {str(e)}"
        })
    # Check if a document with metadata 'source' equal to the filename already exists in the collection.
    existing = await collection.get(where={"source": filename})
    if existing and "ids" in existing and len(existing["ids"]) > 0:
        raise HTTPException(status_code=400, detail="File already added to the collection.")

    split_docs = text_splitter.split_text(content)
    ids = [str(uuid.uuid4()) for doc in split_docs]
    await collection.add(documents=split_docs,
                         ids=ids, metadatas=[{"source": filename} for _ in split_docs])
    results.append({
        "filename": filename,
        "status": "added",
        "chunks_added": len(split_docs),
        "message": "Document added successfully to the knowledge_base collection."
    })

    return {"data": results, "error": False}
