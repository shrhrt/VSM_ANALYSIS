from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import analysis, session

app = FastAPI(title="VSM Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(session.router)


@app.get("/health")
def health():
    return {"status": "ok"}
