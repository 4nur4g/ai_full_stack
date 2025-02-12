from langchain_ollama import OllamaEmbeddings, ChatOllama

MODEL_NAME = "llama3.2"


def get_embedding_model(model="nomic-embed-text"):
    return OllamaEmbeddings(model=model)


def get_llm(model="gemma2:latest"):
    return ChatOllama(model=model)
