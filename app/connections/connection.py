import chromadb
from chromadb import AsyncClientAPI
from fastapi import FastAPI


async def connect_chroma_db(app: FastAPI) -> AsyncClientAPI:
    chroma_client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
    app.state.chroma_client = chroma_client
    if await chroma_client.heartbeat():
        print("Chroma client is connected.")
    else:
        print("Failed to connect to Chroma client.")
    return chroma_client
