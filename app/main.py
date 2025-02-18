from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .connections.connection import connect_chroma_db
from .routers import ai


@asynccontextmanager
async def lifespan(fast_app: FastAPI):
    connect_chroma_db(fast_app)
    yield


app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
