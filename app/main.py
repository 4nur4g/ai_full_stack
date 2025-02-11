from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .connections.connection import connect_chroma_db
from .routers import ai


@asynccontextmanager
async def lifespan(fast_app: FastAPI):
    connect_chroma_db(fast_app)
    yield


app = FastAPI(lifespan=lifespan)

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
