from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from backend.routers import analysis, session

app = FastAPI(title="VSM Analyzer API")

ALLOWED_ORIGINS = [
    "http://localhost:1420",    # Tauri dev (Vite)
    "https://tauri.localhost",  # Tauri 2.x production (Windows/Linux)
    "tauri://localhost",        # Tauri 2.x production (macOS)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_private_network_access_header(request: Request, call_next):
    """Chrome Private Network Access (PNA) プリフライトに応答するミドルウェア。
    https://tauri.localhost → http://localhost:8000 の fetch をブロックされないよう
    Access-Control-Allow-Private-Network ヘッダーを付与する。"""
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "")
        if origin in ALLOWED_ORIGINS:
            headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Private-Network": "true",
            }
            return Response(status_code=204, headers=headers)
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


app.include_router(analysis.router)
app.include_router(session.router)


@app.get("/health")
def health():
    return {"status": "ok"}
