from fastapi import Depends, FastAPI

from .dependencies import get_token_header
from .routers import ai

app = FastAPI(prefix="/api")

app.include_router(
    ai.router,
    prefix="v1/ai",
    tags=["ai"],
    dependencies=[Depends(get_token_header)],
)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}
