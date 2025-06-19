from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from ner_system import LocationNER

app = FastAPI(title="Location NER API",
             description="API for identifying location names and types in text")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ner_system = None

class TextRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    entity: str
    type: str
    source: str
    match: str
    score: int


class NERResponse(BaseModel):
    text: str
    entities: List[Dict[str, Any]]
    status: str = "success"

@app.on_event("startup")
async def startup_event():
    global ner_system
    print("Loading NER system...")
    try:
        ner_system = LocationNER()
        print("NER system loaded successfully")
    except Exception as e:
        print(f"Error loading NER system: {str(e)}")
        raise

@app.get("/")
async def read_root():
    return {"message": "Location NER API is running. Use /analyze endpoint to analyze text."}

@app.post("/analyze", response_model=NERResponse)
async def analyze_text(request: TextRequest):
    if not ner_system:
        raise HTTPException(status_code=503, detail="NER system not initialized")
    
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        entities = ner_system.extract_entities(text)
        
        filtered_entities = []
        for entity in entities:
            if entity['entity'].strip():
                filtered_entities.append({
                    'entity': entity['entity'],
                    'type': entity['type'],
                    'source': entity['source'],
                    'match': entity['match'],
                    'score': entity['score']
                })
        
        return NERResponse(
            text=text,
            entities=filtered_entities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
