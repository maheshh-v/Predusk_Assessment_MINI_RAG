import time
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

app = FastAPI(title='Mini RAG')

# serve static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')

# load pipeline
rag = RAGPipeline()
# print("Pipeline loaded successfully")  # debug

class TextUpload(BaseModel):
    text: str
    doc_id: str = 'user_doc'

class Query(BaseModel):
    query: str
    doc_id: str = 'user_doc'

@app.get('/')
def home():
    static_file = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    return FileResponse(static_file)

# Vercel handler
handler = app

@app.post('/upload')
def upload_text(request: TextUpload):
    try:
        startTime = time.time()
        # print(f"Uploading text: {request.text[:50]}")  # debug
        chunksCount = rag.upsert_text(request.text, request.doc_id)
        processingTime = time.time() - startTime
        
        result = {
            'status': 'success',
            'chunks_stored': chunksCount,
            'processing_time': round(processingTime, 2)
        }
        return result
    except Exception as e:
        # print(f"Error in upload: {e}")  # debug
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/query')
def ask_question(request: Query):
    try:
        start_time = time.time()
        answer, citations = rag.query(request.query, request.doc_id)
        processing_time = time.time() - start_time
        
        # rough cost estimates (very approximate)
        input_tokens = len(request.query.split()) + sum(len(c['source_text'].split()) for c in citations)
        output_tokens = len(answer.split())
        estimated_cost = (input_tokens * 0.00001) + (output_tokens * 0.00002)  # rough groq pricing
        
        response_data = {
            'answer': answer,
            'citations': citations,
            'processing_time': round(processing_time, 2),
            'tokens_used': input_tokens + output_tokens,
            'estimated_cost': f"${estimated_cost:.6f}"
        }
        # print("Query processed successfully")  # debug
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))