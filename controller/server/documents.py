from fastapi import APIRouter

documents_router = APIRouter(prefix="/documents", tags=["documents"])


@documents_router.post("/add_urls")
def add_urls():
    return {"message": "Urls added"}


@documents_router.post("/add_txt")
def add_txt():
    return {"message": "txt added"}


@documents_router.post("/add_pdfs")
def add_pdfs():
    return {"message": "pdfs added"}


@documents_router.post("/add_words")
def add_words():
    return {"message": "words added"}


@documents_router.post("/add_images")
def add_images():
    return {"message": "images added"}
