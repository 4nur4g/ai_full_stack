import os
import tempfile
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Body, File
from fastapi import UploadFile, Request
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from rich.pretty import pprint

from app.utils.graph import is_last_message_with_retrieval
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

    db = request.app.state.chroma_client
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
    db = request.app.state.chroma_client
    llm = get_llm()

    @tool(response_format="content_and_artifact")
    async def retrieve(query: str):
        """Retrieve information related to policy related query."""
        vector_store = Chroma(
            client=db,
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
        )
        retrieved_docs = await vector_store.asimilarity_search(query)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    class State(MessagesState):
        summary: str

    # Define the logic to call the model
    def entry_node(state: State):
        summary = state.get("summary", "")
        messages = state["messages"]
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]

        return {"messages": messages}

    async def summarize_conversation(state: State):

        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]

        class Summary(BaseModel):
            summary: str

        response = await llm.with_structured_output(Summary).ainvoke(messages)
        # Keep track of which messages to delete
        messages_to_delete = []

        messages_to_delete.extend(m.id for m in state["messages"][:-3])
        # Further check if remaining messages relate to tools
        for m in state["messages"][-3:]:
            if (hasattr(m, "tool_calls") and m.tool_calls) or m.type == "tool":
                messages_to_delete.append(m.id)

        # Create the RemoveMessage objects
        delete_messages = [RemoveMessage(id=mid) for mid in messages_to_delete]

        return {"summary": response.summary, "messages": delete_messages}

    # Determine whether to end or summarize the conversation
    def should_summarize(state: State):

        """Return the next node to execute."""

        messages = state["messages"]

        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return True

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    async def query_or_respond(state: State):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system_message_content = (
            "You're Maiyur, a PB Partners employee"
            "You are an Insurance Policy expert"
        )
        if is_last_message_with_retrieval(state):
            system_message_content += " If you're not sure, just say 'I don't know'"
        system_message = SystemMessage(content=system_message_content)
        to_feed = [system_message] + state["messages"]
        response = await llm_with_tools.ainvoke(to_feed)
        return {"messages": [response]}

    def main_node_to_tool_summarize_or_end(state: State) -> Literal["tools", "summarize_conversation", "__end__"]:
        messages_key: str = "messages"
        if isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        if should_summarize(state):
            return "summarize_conversation"
        return END

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    graph_builder = StateGraph(State)
    graph_builder.add_node(entry_node)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(summarize_conversation)

    graph_builder.set_entry_point("entry_node")
    graph_builder.add_edge("entry_node", "query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        main_node_to_tool_summarize_or_end,
    )
    graph_builder.add_edge("tools", "query_or_respond")
    graph_builder.add_edge("summarize_conversation", END)

    async with request.app.state.pool.connection() as connection:
        checkpointer = AsyncPostgresSaver(connection)
        await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)

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
