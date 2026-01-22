# ReliefLens
**Qdrant-powered Disaster Response Assistant**  
Semantic Search + Long-Term Memory + Audio Transcript Support for faster emergency decision-making.

Demo / Documentation

📄 Documentation (Google Doc):
https://docs.google.com/document/d/1QNcVrW30CmGVf1Y2UHSXBY5J5f9XcQmw9RmRX-LZAP0/edit?usp=sharing

It is advised to go through the documentation for a better experience.


## Overview
During disasters (floods, cyclones, earthquakes), responders receive fragmented information across:
- incident reports
- field team updates
- radio/audio communications
- past actions and lessons learned

ReliefLens consolidates these signals using **Qdrant Vector Search**, enabling responders to:
- retrieve relevant evidence instantly
- maintain persistent operational memory
- generate actionable recommendations grounded in retrieved context


## Key Features
- **Incident Semantic Search** using Qdrant vector similarity
- **Hybrid Filtering** by disaster type, state, district, severity
- **Long-term Memory Storage** (decisions + feedback stored and retrievable)
- **Audio Transcript Support** (radio/team comms → embeddings → retrieval)
- **Recommendation Engine** grounded in evidence + memory + audio context


## Tech Stack
- **Backend:** FastAPI (Python)
- **Vector Database:** Qdrant Cloud
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Multimodal Support:** Audio transcript → embedding → retrieval





## Project Structure
Relieflens/
├── main.py
├── memory.py
├── deps.py
├── requirements.txt
├── .gitignore
└── README.md




## Setup Instructions

1) Clone the Repository

```bash
git clone https://github.com/1stAceofSpade/Relieflens.git
cd Relieflens

2) Create and Activate Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

3) Install Dependencies
pip install -r requirements.txt

4) Configure Environment Variables (Optional)

Create a .env file in the project root:

QDRANT_URL=YOUR_QDRANT_CLOUD_URL
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2





If QDRANT_URL and QDRANT_API_KEY are not provided, the backend will attempt to connect to:
http://localhost:6333



Run the Backend
python -m uvicorn main:app --reload --port 8000




Backend URL:
http://127.0.0.1:8000


Swagger UI:
http://127.0.0.1:8000/docs


API Usage Examples
Health Check
curl http://127.0.0.1:8000/health



Ingest Incident
curl -X POST "http://127.0.0.1:8000/ingest" \
-H "Content-Type: application/json" \
-d '{
  "type": "report",
  "disaster_type": "flood",
  "title": "Flooding in low-lying area",
  "text": "Heavy rainfall caused severe flooding. People stranded near river bank. Power outage reported.",
  "location": {"state": "Odisha", "district": "Cuttack"},
  "severity": 4,
  "source": "synthetic",
  "timestamp": "2026-01-21T10:00:00Z"
}'

Search Incidents (Hybrid Filter + Semantic Search)
curl -X POST "http://127.0.0.1:8000/search" \
-H "Content-Type: application/json" \
-d '{
  "query": "people stranded flood power outage",
  "top_k": 5,
  "filters": {"disaster_type": "flood", "state": "Odisha"}
}'

Add Memory
curl -X POST "http://127.0.0.1:8000/memory/add" \
-H "Content-Type: application/json" \
-d '{
  "user_id": "demo_user",
  "incident_id": null,
  "memory_type": "decision",
  "text": "Evacuation started. Need 2 more boats near river bank. Power lines down: electrocution risk.",
  "timestamp": "2026-01-21T18:00:00Z"
}'

Ingest Audio Transcript (Multimodal)
curl -X POST "http://127.0.0.1:8000/audio/ingest" \
-H "Content-Type: application/json" \
-d '{
  "user_id": "demo_user",
  "incident_id": null,
  "audio_source": "radio",
  "transcript": "Team Alpha: two families stranded near river bank. Water rising fast. Power lines down. Need 2 boats urgently.",
  "language": "en",
  "timestamp": "2026-01-21T19:30:00Z"
}'

Search Audio Transcripts
curl -X POST "http://127.0.0.1:8000/audio/search" \
-H "Content-Type: application/json" \
-d '{
  "query": "need boats stranded river power lines down",
  "top_k": 5,
  "user_id": "demo_user"
}'

Get Recommendations (Evidence + Memory + Audio)
curl -X POST "http://127.0.0.1:8000/recommend" \
-H "Content-Type: application/json" \
-d '{
  "user_id": "demo_user",
  "query": "What should responders do next for this flood situation in Odisha?",
  "top_k": 5,
  "filters": {"disaster_type": "flood", "state": "Odisha"}
}'


Why Qdrant is Core to ReliefLens?

ReliefLens uses Qdrant not only for storage, but for:

semantic retrieval of incident reports

persistent operational memory (decisions + outcomes)

multimodal retrieval of audio transcripts

hybrid search combining metadata filters + similarity search

Future Improvements

Real audio upload + transcription (Whisper / Gemini)

Image ingestion using multimodal embeddings (CLIP)

Improved risk scoring + duplicate recommendation reduction
