from fastapi import APIRouter, WebSocket, WebSocketDisconnect

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("")
def chat():
    return {"message": "Hello World"}


@chat_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            message = await websocket.receive_json()
            await websocket.send_json(message)
        except WebSocketDisconnect:
            break
    await websocket.close()
