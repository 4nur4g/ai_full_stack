import chromadb
from chromadb import ClientAPI
from fastapi import FastAPI
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings


def connect_chroma_db(app: FastAPI) -> ClientAPI:
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    app.chroma_client = chroma_client
    if chroma_client.heartbeat():
        print("Chroma client is connected.")
    else:
        print("Failed to connect to Chroma client.")
    return chroma_client


DB_URI = (
    f"postgresql://{settings.psql_username}:{settings.psql_password}@"
    f"{settings.psql_host}:{settings.psql_port}/{settings.psql_database}?sslmode={settings.psql_sslmode}"
)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}


async def connect_pg(app: FastAPI):
    app.pool = AsyncConnectionPool(
        # Example configuration
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )
    return AsyncConnectionPool(
        # Example configuration
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )
