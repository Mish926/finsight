"""
FinSight FastAPI Server

Session isolation: every visitor gets their own cookie-based session, and
therefore their own isolated document index. What one visitor uploads is
never visible to, searchable by, or deletable by another -- this was NOT
true in earlier versions of this app (a single global index was shared
by every visitor), and is required for any genuinely public deployment.
"""

import os
import re
import shutil
import secrets
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.pipeline import FinSightPipeline

app = FastAPI(title="FinSight", version="2.0.0")

SESSION_COOKIE_NAME = "finsight_session"
SESSION_TTL_HOURS = 24  # sessions inactive longer than this are cleaned up
SESSION_ROOT = Path("data/sessions")
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

# In-memory registry: session_id -> {"pipeline": FinSightPipeline, "last_active": datetime}
# This is intentionally simple (not a database) -- fine for a single-process
# deployment. On restart, in-memory state is lost, but each session's
# on-disk index (if the host has persistent storage) can be reloaded
# transparently the next time that visitor's cookie comes back.
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
    """FastAPI dependency: returns a session_id, creating a new one (and
    setting a cookie) if this visitor doesn't have one yet, or if the
    cookie value doesn't look like a session ID we'd have issued."""
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
    """Background task: periodically removes sessions (both in-memory and
    their on-disk data) that have been inactive longer than SESSION_TTL_HOURS,
    so a public deployment doesn't accumulate unlimited abandoned indexes."""
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


@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(_cleanup_stale_sessions())


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
    """Remove one document from THIS session's index only -- does not
    touch the underlying PDF file on disk, and cannot affect any other
    visitor's session."""
    try:
        result = pipeline.remove_document(filename)
        return JSONResponse(content={"success": True, **result})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
async def clear_index(session_id: str = Depends(get_session)):
    """Clears THIS session's search index only -- does not touch any
    files on disk, and cannot affect any other visitor's session."""
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
