import chromadb
from chromadb import ClientAPI
from fastapi import FastAPI


def connect_chroma_db(app: FastAPI) -> ClientAPI:
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    app.state.chroma_client = chroma_client
    if chroma_client.heartbeat():
        print("Chroma client is connected.")
    else:
        print("Failed to connect to Chroma client.")
    return chroma_client
