from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings

MODEL_NAME = "llama3.2"


def get_embedding_model(model="nomic-embed-text"):
    return OllamaEmbeddings(model=model)


def get_llm(model="llama3.2:latest"):
    # return ChatOllama(model=model)
    return ChatOpenAI(api_key=settings.openai_api_key, streaming=True)
