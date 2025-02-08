from fastapi import APIRouter

router = APIRouter()

fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}
@router.get("/")
async def read_items():
    return fake_items_db
