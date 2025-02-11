from typing import Annotated

from chromadb import HttpClient
from fastapi import APIRouter, HTTPException, Body
from fastapi import UploadFile, Request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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
    vector_store = Chroma(
        client=db,
        collection_name="knowledge_base_v1",
        embedding_function=get_embedding_model(),
    )
    existing = vector_store.get(where={"source": file.filename})
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


@router.post("/chat")
async def chat(body: Annotated[Chat, Body()], request: Request):
    """
    Accepts a message in the request body and returns a response from the LLM using a RetrievalQA chain.
    """
    db = request.app.state.chroma_client

    vector_store = Chroma(
        client=db,
        collection_name="knowledge_base_v1",
        embedding_function=get_embedding_model(),
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    system_prompt = (
        "Use only the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Strictly use the context to answer the question. "
        "Use three sentence maximum and keep the answer concise. \n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    result = chain.invoke({"input": body.message})

    return {"data": result["answer"], "error": False}
