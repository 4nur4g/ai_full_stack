from typing import Literal

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from app.utils.graph import is_last_message_with_retrieval


async def get_graph(llm, embedding_model, db, collection_name, request):
    @tool(response_format="content_and_artifact")
    async def retrieve(query: str):
        """Retrieve information related to policy related query."""
        vector_store = Chroma(
            client=db,
            collection_name=collection_name,
            embedding_function=embedding_model,
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

    async with request.app.pool.connection() as connection:
        checkpointer = AsyncPostgresSaver(connection)
        await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph
