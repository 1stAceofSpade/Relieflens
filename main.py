"""
ReliefLens FastAPI Backend
Qdrant-powered Disaster Response Assistant
(Search + Memory + Recommendations + Traceability)
"""

import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any


from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    PayloadSchemaType,
)

from sentence_transformers import SentenceTransformer

load_dotenv()

# ----------------------------
# Config
# ----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

INCIDENTS_COLLECTION = "incidents"
MEMORY_COLLECTION = "memory"
AUDIO_COLLECTION = "audio_reports"

EMBED_DIM = 384
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="ReliefLens Backend",
    description="Qdrant-powered Disaster Response (Search + Memory + Recommendations)",
    version="1.0.0",
)

# ----------------------------
# Clients
# ----------------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def embed_text(text: str) -> List[float]:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty for embedding.")
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def ensure_collections() -> None:
    collections = [
        INCIDENTS_COLLECTION,
        MEMORY_COLLECTION,
        AUDIO_COLLECTION,
    ]

    for name in collections:
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )


def ensure_indexes() -> None:
    # Incidents
    for key in ["disaster_type", "state", "district", "type", "source"]:
        try:
            client.create_payload_index(
                collection_name=INCIDENTS_COLLECTION,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    try:
        client.create_payload_index(
            collection_name=INCIDENTS_COLLECTION,
            field_name="severity",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass

    # Memory
    for key in ["user_id", "incident_id", "memory_type", "dedupe_key"]:
        try:
            client.create_payload_index(
                collection_name=MEMORY_COLLECTION,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    # Audio
    for key in ["user_id", "incident_id", "audio_source"]:
        try:
            client.create_payload_index(
                collection_name=AUDIO_COLLECTION,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    ensure_collections()
    ensure_indexes()


# ----------------------------
# Schemas
# ----------------------------
class Location(BaseModel):
    state: str
    district: str


class IngestRequest(BaseModel):
    type: str = Field(..., description="report|alert|transcript|image")
    disaster_type: str = Field(..., description="flood|earthquake|wildfire|cyclone")
    title: str
    text: str
    location: Location
    severity: int = Field(..., ge=1, le=5)
    source: str = Field(..., description="news|ngo|user|synthetic")
    timestamp: Optional[str] = None


class SearchFilters(BaseModel):
    disaster_type: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    severity_min: Optional[int] = Field(None, ge=1, le=5)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[SearchFilters] = None


class MemoryAddRequest(BaseModel):
    user_id: str
    incident_id: Optional[str] = None
    memory_type: str = Field(..., description="timeline|decision|feedback")
    text: str
    timestamp: Optional[str] = None
    dedupe_key: Optional[str] = None


class MemoryQueryRequest(BaseModel):
    user_id: str
    incident_id: Optional[str] = None
    query: str
    top_k: int = Field(5, ge=1, le=20)


class RecommendRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(6, ge=1, le=20)
    filters: Optional[SearchFilters] = None


def build_incident_filter(filters: Optional[SearchFilters]) -> Optional[Filter]:
    if not filters:
        return None

    must_conditions = []

    if filters.disaster_type:
        must_conditions.append(
            FieldCondition(key="disaster_type", match=MatchValue(value=filters.disaster_type))
        )

    if filters.state:
        must_conditions.append(
            FieldCondition(key="state", match=MatchValue(value=filters.state))
        )

    if filters.district:
        must_conditions.append(
            FieldCondition(key="district", match=MatchValue(value=filters.district))
        )

    if filters.severity_min is not None:
        must_conditions.append(
            FieldCondition(key="severity", range=Range(gte=int(filters.severity_min)))
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


def build_memory_filter(user_id: str, incident_id: Optional[str]) -> Filter:
    must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]

    if incident_id:
        must_conditions.append(
            FieldCondition(key="incident_id", match=MatchValue(value=incident_id))
        )

    return Filter(must=must_conditions)
import re
from typing import Optional

def hybrid_rerank(query: str, results: list[dict], boost_fields: Optional[list[str]] = None) -> list[dict]:
    if not results:
        return results

    boost_fields = boost_fields or ["title", "text", "state", "district", "disaster_type", "transcript"]

    q = query.lower().strip()
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", q) if len(t) >= 3]
    if not tokens:
        return results

    ops_keywords = {"evac", "evacuation", "boat", "boats", "rescue", "power", "electrocution", "shelter", "triage"}

    def score_boost(payload: dict) -> float:
        blob_parts = []
        for f in boost_fields:
            val = payload.get(f)
            if isinstance(val, str):
                blob_parts.append(val.lower())
        blob = " ".join(blob_parts)

        boost = 0.0
        for t in tokens:
            if t in blob:
                boost += 0.02

        for k in ops_keywords:
            if k in blob and k in q:
                boost += 0.05

        return boost

    reranked = []
    for r in results:
        payload = r.get("payload", {}) or {}
        base = float(r.get("score", 0.0))
        new_score = base + score_boost(payload)
        rr = dict(r)
        rr["score"] = new_score
        reranked.append(rr)

    reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reranked


@app.get("/health")
async def health():
    return {"status": "operational", "time": now_iso(), "qdrant_url": QDRANT_URL}


@app.post("/ingest")
async def ingest_incident(req: IngestRequest):
    incident_id = str(uuid.uuid4())
    ts = req.timestamp or now_iso()

    payload = req.dict()
    payload["timestamp"] = ts

    payload["state"] = req.location.state
    payload["district"] = req.location.district

    vector = embed_text(req.text)

    client.upsert(
        collection_name=INCIDENTS_COLLECTION,
        points=[
            PointStruct(
                id=incident_id,
                vector=vector,
                payload=payload,
            )
        ],
    )

    return {"status": "success", "incident_id": incident_id}


import re

def hybrid_rerank(query: str, results: list[dict], boost_fields: list[str] | None = None) -> list[dict]:
    if not results:
        return results

    boost_fields = boost_fields or ["title", "text", "state", "district", "disaster_type", "transcript"]

    q = query.lower().strip()
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", q) if len(t) >= 3]
    if not tokens:
        return results

    def score_boost(payload: dict) -> float:
        blob_parts = []
        for f in boost_fields:
            val = payload.get(f)
            if isinstance(val, str):
                blob_parts.append(val.lower())
        blob = " ".join(blob_parts)

        boost = 0.0
        for t in tokens:
            if t in blob:
                boost += 0.02

        ops_keywords = ["evac", "evacuation", "boat", "boats", "rescue", "power", "electrocution", "shelter", "triage"]
        for k in ops_keywords:
            if k in blob and k in q:
                boost += 0.05

        return boost

    reranked = []
    for r in results:
        payload = r.get("payload", {}) or {}
        base = float(r.get("score", 0.0))
        rr = dict(r)
        rr["score"] = base + score_boost(payload)
        reranked.append(rr)

    reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reranked



@app.post("/search")
async def search_incidents(req: SearchRequest):
    vector = embed_text(req.query)
    query_filter = build_incident_filter(req.filters)

    results = client.search(
        collection_name=INCIDENTS_COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=req.top_k,
        with_payload=True,
    )

    out = [
        {"id": r.id, "score": float(r.score), "payload": r.payload}
        for r in results
    ]

    out = hybrid_rerank(
        req.query,
        out,
        boost_fields=["title", "text", "state", "district", "disaster_type"]
    )

    return out




@app.post("/memory/add")
async def memory_add(req: MemoryAddRequest):
    mem_id = str(uuid.uuid4())
    ts = req.timestamp or now_iso()

    # Dedupe protection (optional)
    if req.dedupe_key:
        must = [
            FieldCondition(key="dedupe_key", match=MatchValue(value=req.dedupe_key)),
            FieldCondition(key="user_id", match=MatchValue(value=req.user_id)),
        ]
        if req.incident_id:
            must.append(FieldCondition(key="incident_id", match=MatchValue(value=req.incident_id)))

        existing = client.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=1,
            with_payload=True,
        )

        points = existing[0]
        if points:
            return {"status": "skipped_duplicate", "memory_id": str(points[0].id)}

    payload = {
        "user_id": req.user_id,
        "incident_id": req.incident_id,
        "memory_type": req.memory_type,
        "text": req.text,
        "timestamp": ts,
        "dedupe_key": req.dedupe_key,
    }

    vector = embed_text(req.text)

    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=mem_id, vector=vector, payload=payload)],
    )

    return {"status": "success", "memory_id": mem_id}


@app.post("/memory/query")
async def memory_query(req: MemoryQueryRequest):
    vector = embed_text(req.query)
    query_filter = build_memory_filter(req.user_id, req.incident_id)

    results = client.search(
        collection_name=MEMORY_COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=req.top_k,
        with_payload=True,
    )

    return [{"id": r.id, "score": float(r.score), "payload": r.payload} for r in results]

def rule_based_recommendations(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not evidence:
        return {
            "risk_level": "UNKNOWN",
            "recommendations": [
                {
                    "priority": 1,
                    "action": "Insufficient evidence to recommend specific actions.",
                    "why": "No relevant incident evidence was retrieved from the database.",
                }
            ],
        }

    max_sev = 1
    disaster_type = None

    for e in evidence:
        sev = e.get("payload", {}).get("severity", 1)
        max_sev = max(max_sev, int(sev))
        if not disaster_type:
            disaster_type = e.get("payload", {}).get("disaster_type")

    if max_sev >= 4:
        risk_level = "HIGH"
    elif max_sev == 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    base_actions = {
        "flood": [
            "Issue evacuation advisory for low-lying areas and move people to shelters.",
            "Deploy boats/rescue teams to reported stranded locations.",
            "Ensure clean drinking water and prevent electrocution hazards.",
        ],
        "earthquake": [
            "Check for structural damage and restrict access to unsafe buildings.",
            "Deploy medical triage units and search-and-rescue teams.",
            "Prepare aftershock safety alerts and temporary shelters.",
        ],
        "wildfire": [
            "Evacuate downwind zones and establish firebreaks.",
            "Deploy firefighting units and monitor air quality.",
            "Set up medical support for smoke inhalation cases.",
        ],
        "cyclone": [
            "Issue cyclone shelter routing and evacuate coastal areas.",
            "Secure power lines and critical infrastructure.",
            "Stage medical + food supplies for post-landfall response.",
        ],
    }

    actions = base_actions.get(
        disaster_type,
        [
            "Verify reports and prioritize life-threatening situations.",
            "Coordinate rescue and medical response teams.",
            "Provide public safety guidance and shelter info.",
        ],
    )

    recs = [
        {
            "priority": i + 1,
            "action": act,
            "why": f"Grounded in retrieved {disaster_type or 'disaster'} evidence.",
        }
        for i, act in enumerate(actions[:3])
    ]

    return {"risk_level": risk_level, "recommendations": recs}

def qdrant_audio_search(query: str, top_k: int = 3, user_id: Optional[str] = None):
    vector = embed_text(query)

    conditions = []
    if user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=AUDIO_COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    out = [
        {"id": r.id, "score": float(r.score), "payload": r.payload}
        for r in results
    ]

    out = hybrid_rerank(
        query,
        out,
        boost_fields=["transcript", "audio_source", "language", "user_id", "incident_id"]
    )

    return out


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    incident_results = await search_incidents(
        SearchRequest(query=req.query, top_k=req.top_k, filters=req.filters)
    )

    primary_incident_id = incident_results[0]["id"] if incident_results else None

    mem_results = await memory_query(
        MemoryQueryRequest(
            user_id=req.user_id,
            incident_id=primary_incident_id,
            query=req.query,
            top_k=5,
        )
    )

    audio_results = qdrant_audio_search(
        query=req.query,
        top_k=3,
        user_id=req.user_id,
    )

    rec_out = rule_based_recommendations(incident_results)
    risk_level = rec_out["risk_level"]

    mem_text_all = " ".join([m["payload"].get("text", "") for m in mem_results]).lower()
    evacuation_started = ("evacuation started" in mem_text_all) or ("evacuated" in mem_text_all)
    boats_needed = ("boat" in mem_text_all) or ("boats" in mem_text_all)
    power_risk = ("power line" in mem_text_all) or ("electrocution" in mem_text_all) or ("power outage" in mem_text_all)

    recommendations_ics = []

    if evacuation_started:
        recommendations_ics.append("Scale evacuation: expand coverage, track shelter capacity, prevent re-entry.")
    else:
        recommendations_ics.append("Start evacuation of low-lying zones and move residents to verified shelters.")

    recommendations_ics.append("Deploy rescue teams to stranded locations; prioritize children/elderly/medical emergencies.")

    if boats_needed:
        recommendations_ics.append("Request + stage 2 more boats/rafts and fuel near river bank access points.")
    else:
        recommendations_ics.append("Stage boats/ropes/life jackets at key entry points for rapid deployment.")

    recommendations_ics.append("Distribute clean water + ORS + chlorine tablets + dry food packets.")

    if power_risk:
        recommendations_ics.append("Cordon off downed power lines + cut power in flooded zones + coordinate with electricity dept.")
    else:
        recommendations_ics.append("Warn about electrocution risk + unsafe structures; use PPE and avoid fast-moving water.")

    recommendations_ics.append("Broadcast evacuation routes, shelter locations, and helpline numbers via SMS/PA/local radio.")
    recommendations_ics.append("Set up triage at shelters; monitor injuries, hypothermia, diarrheal illness.")

    rec_text = (
        f"Query: {req.query}\n"
        f"Incident: {primary_incident_id}\n"
        f"Risk: {risk_level}\n"
        f"Recs: " + "; ".join(recommendations_ics[:8])
    )

    should_store = True

    # Deduping using memory_query similarity (works fine for hackathon)
    recent_mem = await memory_query(
        MemoryQueryRequest(
            user_id=req.user_id,
            incident_id=primary_incident_id,
            query=rec_text,
            top_k=1,
        )
    )
    if recent_mem and float(recent_mem[0]["score"]) > 0.92:
        should_store = False

    if should_store:
        await memory_add(
            MemoryAddRequest(
                user_id=req.user_id,
                incident_id=primary_incident_id,
                memory_type="decision",
                text=rec_text,
            )
        )

    return {
        "risk_level": risk_level,
        "recommendations": rec_out["recommendations"],
        "recommendations_ics": recommendations_ics,
        "evidence_used": incident_results,
        "memory_used": mem_results,
        "audio_used": audio_results,
        "stored_new_memory": should_store,
    }


class AudioIngestRequest(BaseModel):
    user_id: str
    incident_id: Optional[str] = None
    audio_source: str = "voice_note"
    transcript: str
    language: Optional[str] = "en"
    timestamp: Optional[str] = None


class AudioSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: Optional[str] = None
    incident_id: Optional[str] = None


@app.post("/audio/ingest")
async def audio_ingest(req: AudioIngestRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    vector = embed_text(req.transcript)

    payload = {
        "type": "audio",
        "user_id": req.user_id,
        "incident_id": req.incident_id,
        "audio_source": req.audio_source,
        "transcript": req.transcript,
        "language": req.language,
        "timestamp": req.timestamp or datetime.now(timezone.utc).isoformat(),
    }

    audio_id = str(uuid.uuid4())

    client.upsert(
        collection_name=AUDIO_COLLECTION,
        points=[PointStruct(id=audio_id, vector=vector, payload=payload)],
    )

    return {"status": "success", "audio_id": audio_id}


@app.post("/audio/search")
async def audio_search(req: AudioSearchRequest):
    vector = embed_text(req.query)

    conditions = []
    if req.user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=req.user_id)))
    if req.incident_id:
        conditions.append(FieldCondition(key="incident_id", match=MatchValue(value=req.incident_id)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=AUDIO_COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=req.top_k,
        with_payload=True,
    )

    return [
        {"id": r.id, "score": float(r.score), "payload": r.payload}
        for r in results
    ]
