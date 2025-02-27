from langchain_core.messages import BaseMessage


def is_last_message_with_retrieval(state):
    """
    Checks if the last message in the conversation state is from a tool.
    """
    if not state["messages"]:
        return False  # No messages in state
    last_message: BaseMessage = state["messages"][-1]
    return last_message.type == "tool" and last_message.name == "retrieve" and last_message.content

def save_graph_image(graph):
    """
    Saves the graph image to the current directory.
    """
    graph_image = graph.get_graph().draw_mermaid_png()
    # Save the image to the current directory
    image_path = "graph.png"
    with open(image_path, "wb") as f:
        f.write(graph_image)
    print(f"Graph saved as {image_path}")
