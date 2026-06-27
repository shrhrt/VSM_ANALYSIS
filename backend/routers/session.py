import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("/env")
async def get_env():
    """Return OneDrive environment variables for cross-PC path resolution."""
    return {
        "onedrive_commercial": os.environ.get("OneDriveCommercial", ""),
        "onedrive": os.environ.get("OneDrive", ""),
    }


class EntryMeta(BaseModel):
    filename:     str = ""
    absolutePath: str = ""
    relativePath: str = ""
    onedrivePath: str = ""


class ResolveRequest(BaseModel):
    session_path: str
    entries: list[EntryMeta]


class ResolvedEntry(BaseModel):
    filename:      str
    resolved_path: str


class ResolveResponse(BaseModel):
    resolved: list[ResolvedEntry]
    missing:  list[str]


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_paths(req: ResolveRequest) -> ResolveResponse:
    """
    Resolve file paths for session loading using multiple fallback strategies:
    1. Relative path from session file directory
    2. Absolute path (same machine)
    3. Same directory as session file (filename only)
    4. OneDrive cross-PC path (OneDriveCommercial / OneDrive env vars)
    """
    session_dir = Path(req.session_path).parent

    od_roots: list[Path] = []
    for key in ("OneDriveCommercial", "OneDrive"):
        v = os.environ.get(key, "")
        if v:
            od_roots.append(Path(v))

    resolved: list[ResolvedEntry] = []
    missing: list[str] = []

    for entry in req.entries:
        path: Optional[Path] = None

        # 1. Relative path from session directory
        if entry.relativePath and not path:
            try:
                candidate = (session_dir / entry.relativePath).resolve()
                if candidate.exists():
                    path = candidate
            except Exception:
                pass

        # 2. Absolute path (same machine)
        if entry.absolutePath and not path:
            candidate = Path(entry.absolutePath)
            if candidate.exists():
                path = candidate

        # 3. Same directory as session file
        if entry.filename and not path:
            candidate = session_dir / entry.filename
            if candidate.exists():
                path = candidate

        # 4. OneDrive cross-PC path
        if entry.onedrivePath and not path:
            for od_root in od_roots:
                candidate = od_root / entry.onedrivePath
                if candidate.exists():
                    path = candidate
                    break

        if path:
            resolved.append(ResolvedEntry(filename=entry.filename, resolved_path=str(path)))
        else:
            label = entry.filename or entry.absolutePath or entry.relativePath or "(unknown)"
            missing.append(label)

    return ResolveResponse(resolved=resolved, missing=missing)
