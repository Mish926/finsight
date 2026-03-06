"""
FinSight FastAPI Server
"""

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.pipeline import FinSightPipeline

app = FastAPI(title="FinSight", version="1.0.0")

pipeline = FinSightPipeline(index_dir="data/index")

PDF_DIR = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("api/templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_path.read_text())


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = PDF_DIR / file.filename
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
async def query(request: QueryRequest):
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
async def stats():
    return JSONResponse(content=pipeline.get_stats())


@app.delete("/index")
async def clear_index():
    import glob
    for f in glob.glob("data/index/*"):
        os.remove(f)
    global pipeline
    pipeline = FinSightPipeline(index_dir="data/index")
    return JSONResponse(content={"success": True, "message": "Index cleared"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=5002, reload=False)
