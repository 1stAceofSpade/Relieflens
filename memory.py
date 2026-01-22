from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# We will import shared objects from a "deps" module (you'll create it)


router = APIRouter(prefix="/memory", tags=["Memory"])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryAddRequest(BaseModel):
    user_id: str
    incident_id: Optional[str] = None
    memory_type: str = Field(..., description="timeline|decision|feedback")
    text: str
    timestamp: Optional[str] = None


class MemoryQueryRequest(BaseModel):
    user_id: str
    incident_id: Optional[str] = None
    query: str
    top_k: int = Field(5, ge=1, le=20)


@router.post("/add")
async def add_memory(req: MemoryAddRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Memory text cannot be empty.")

    vector = embed_text(req.text)
    mem_id = str(uuid.uuid4())

    payload = {
        "user_id": req.user_id,
        "incident_id": req.incident_id,
        "memory_type": req.memory_type,
        "text": req.text,
        "timestamp": req.timestamp or now_iso(),
    }

    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=mem_id, vector=vector, payload=payload)],
    )

    return {"status": "success", "memory_id": mem_id}


@router.post("/query")
async def query_memory(req: MemoryQueryRequest):
    vector = embed_text(req.query)

    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=req.user_id))
    ]
    if req.incident_id:
        must_conditions.append(
            FieldCondition(key="incident_id", match=MatchValue(value=req.incident_id))
        )

    results = client.search(
        collection_name=MEMORY_COLLECTION,
        query_vector=vector,
        query_filter=Filter(must=must_conditions),
        limit=req.top_k,
        with_payload=True,
    )

    return [
        {"id": r.id, "score": float(r.score), "payload": r.payload}
        for r in results
    ]
