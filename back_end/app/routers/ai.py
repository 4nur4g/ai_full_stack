from io import BytesIO
from typing import Annotated

from app.utils.llm import get_embedding_model, get_llm
from app.utils.loader import BytesIOPyMuPDFLoader
from fastapi import APIRouter, HTTPException, Body, File
from fastapi import UploadFile, Request
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from rich.pretty import pprint

router = APIRouter()


@router.post("/upload-files", summary="Upload one or more text files to the knowledge_base collection")
async def upload_files(
        request: Request,
        files: list[UploadFile] = File(...),
):
    """
    Accepts one or more text files via form-data (field name 'files'),
    checks if each document (using its filename as a unique identifier in metadata 'source')
    is already in the Chroma vector store (collection: 'knowledge_base_v2'),
    and adds it if not.
    """
    if not (1 <= len(files) <= 5):
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 files.")

    db = request.app.state.chroma_client
    vector_store = Chroma(
        client=db,
        collection_name="knowledge_base_v2",
        embedding_function=get_embedding_model(),
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

        # Process file based on its content type.
        if file.content_type == "text/plain":
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Decoding error: {str(e)}"
                })
                continue
            documents = [Document(page_content=content, metadata={"source": file.filename})]
        elif file.content_type == "application/pdf":
            pdf_stream = BytesIO(content_bytes)
            try:
                loader = BytesIOPyMuPDFLoader(pdf_stream)
                documents = loader.load()
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Error processing PDF: {str(e)}"
                })
                continue
        else:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported file type: {file.content_type}"
            })
            continue

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
    db = request.app.state.chroma_client
    llm = get_llm()

    @tool(response_format="content_and_artifact")
    async def retrieve(query: str):
        """Retrieve information related to a query."""
        vector_store = Chroma(
            client=db,
            collection_name="knowledge_base_v2",
            embedding_function=get_embedding_model(),
        )
        retrieved_docs = await vector_store.asimilarity_search(query, k=3)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    async def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system_message_content = (
            "You're Maiyur, a PB Partners employee"
            "You are a Insurance Policy expert"
        )
        system_message = SystemMessage(content=system_message_content)
        response = await llm_with_tools.ainvoke([system_message] + state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    # Step 3: Generate a response using the retrieved content.
    async def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
               or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = await llm.ainvoke(prompt)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    if not body.stream:
        result = await graph.ainvoke({"messages": [{"role": "user", "content": body.message}]})
        pprint(result)
        return {"data": result["messages"][-1].content, "error": False}

    async def event_generator():
        async for message, metadata in graph.astream(
                {"messages": [{"role": "user", "content": body.message}]},
                stream_mode="messages",
        ):
            if metadata["langgraph_node"] in ["query_or_respond", "generate"]:
                content = message.content
                yield f"data: {content}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
