from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .connections.connection import connect_chroma_db, connect_pg
from .routers import ai


@asynccontextmanager
async def lifespan(fast_app: FastAPI):
    connect_chroma_db(fast_app)
    pg_async_connection_pool = await connect_pg(fast_app)
    yield
    pg_async_connection_pool.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(
    ai.router,
    prefix="/ai",
    tags=["ai"],
)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


def start():
    """To run the development server"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=3006, reload=True)


if __name__ == "__main__":
    start()
