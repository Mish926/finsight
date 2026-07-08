"""
FinSight FastAPI Server

Session isolation: every visitor gets their own cookie-based session, and
therefore their own isolated document index. What one visitor uploads is
never visible to, searchable by, or deletable by another.

sys.path is set up explicitly at the top so this file finds `core` and
`agents` regardless of how it's launched (bare `python api/app.py`,
`PYTHONPATH=. python api/app.py`, or a hosting platform's own start
command) -- relying on the launch command alone to set PYTHONPATH
correctly is exactly the kind of thing that silently breaks on a
platform you haven't tested on.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import shutil
import secrets
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.pipeline import FinSightPipeline

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_cleanup_stale_sessions())
    yield


app = FastAPI(title="FinSight", version="2.0.0", lifespan=lifespan)

SESSION_COOKIE_NAME = "finsight_session"
SESSION_TTL_HOURS = 24
SESSION_ROOT = Path("data/sessions")
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

sessions: dict = {}


class QueryRequest(BaseModel):
    question: str


def _session_index_dir(session_id: str) -> str:
    return str(SESSION_ROOT / session_id / "index")


def _session_pdf_dir(session_id: str) -> Path:
    d = SESSION_ROOT / session_id / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_session(
    response: Response,
    finsight_session: Optional[str] = Cookie(default=None)
) -> str:
    session_id = finsight_session
    if not session_id or not re.match(r'^[A-Za-z0-9_-]{16,64}$', session_id):
        session_id = secrets.token_urlsafe(32)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=SESSION_TTL_HOURS * 3600,
            httponly=True,
            samesite="lax"
        )

    if session_id not in sessions:
        sessions[session_id] = {
            "pipeline": FinSightPipeline(index_dir=_session_index_dir(session_id)),
            "last_active": datetime.utcnow()
        }
    else:
        sessions[session_id]["last_active"] = datetime.utcnow()

    return session_id


def get_pipeline(session_id: str = Depends(get_session)) -> FinSightPipeline:
    return sessions[session_id]["pipeline"]


async def _cleanup_stale_sessions():
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
        stale_ids = [sid for sid, data in sessions.items() if data["last_active"] < cutoff]
        for sid in stale_ids:
            del sessions[sid]
            session_dir = SESSION_ROOT / sid
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
            print(f"[Cleanup] Removed stale session {sid}")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("api/templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_path.read_text())


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session),
    pipeline: FinSightPipeline = Depends(get_pipeline)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_dir = _session_pdf_dir(session_id)
    save_path = pdf_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = pipeline.index_document(str(save_path))
        return JSONResponse(content={
            "success": True,
            "message": f"Indexed {result['chunks']} chunks from {result['filename']}",
            "filename": result["filename"],
            "chunks": result["chunks"],
            "total_indexed": result["total_indexed"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest, pipeline: FinSightPipeline = Depends(get_pipeline)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = pipeline.query(request.question)
        if "error" in result and result["answer"] is None:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats(pipeline: FinSightPipeline = Depends(get_pipeline)):
    return JSONResponse(content=pipeline.get_stats())


@app.delete("/documents/{filename}")
async def delete_document(filename: str, pipeline: FinSightPipeline = Depends(get_pipeline)):
    try:
        result = pipeline.remove_document(filename)
        return JSONResponse(content={"success": True, **result})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
async def clear_index(session_id: str = Depends(get_session)):
    import glob
    idx_dir = _session_index_dir(session_id)
    for f in glob.glob(f"{idx_dir}/*"):
        os.remove(f)
    sessions[session_id]["pipeline"] = FinSightPipeline(index_dir=idx_dir)
    return JSONResponse(content={"success": True, "message": "Index cleared. Uploaded PDF files were left untouched on disk."})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5002))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)
