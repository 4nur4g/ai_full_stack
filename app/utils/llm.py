from langchain_ollama import OllamaEmbeddings, ChatOllama

EMBEDDING_MODEL = "nomic-embed-text"
MODEL_NAME = "llama3.2"
def get_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
def get_llm():
    return ChatOllama(model=MODEL_NAME)
