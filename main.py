from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from controller.server.chat import chat_router
from controller.server.documents import documents_router

app = FastAPI(
    title="智能体",
    description="这是一个关于智能体API的文档",
    version="1.0.0",
    contact={
        "name": "huangrx",
        "mobile": "15236325327",
    }
)

app.include_router(chat_router)
app.include_router(documents_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
