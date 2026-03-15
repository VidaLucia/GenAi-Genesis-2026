# SCOPE Fraud Detection Simulator

SCOPE is an interactive fraud-risk simulation platform that combines machine learning signals with behavioral and geolocation heuristics. It scores card transactions in real time, explains why a transaction was flagged, and provides an analyst-friendly history of decisions.

## Project Overview

This project includes:
- A FastAPI backend that calculates fraud probability from transaction, profile, and location context.
- An ensemble model pipeline (Naive Bayes, Decision Tree, Random Forest) with weighted scoring.
- Explainable output including risk label, score breakdown, and human-readable reasons.
- A Vite + vanilla JavaScript frontend with map-based phone location input and heatmap visualization.
- Transaction logging endpoints for analyst review and transaction history.

## Repository Structure

- `backend/` - API, services, model loading, and training scripts
- `frontend/` - user interface (Vite app)
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Setup Instructions

### 1) Clone and open project

```bash
git clone https://github.com/VidaLucia/GenAi-Genesis-2026.git
cd GenAi-Genesis-2026
```

### 2) Backend setup

From the project root:

```bash
python -m venv venv
```

Activate the virtual environment:

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start API server:

```bash
cd backend
python run.py
```

Backend runs at:
- `http://localhost:8000`
- API root check: `GET /`

### 3) Frontend setup

Open a new terminal from the project root:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
- `http://localhost:5173`

## Optional: Retrain Models and Profiles

If you want to regenerate model files and user profiles:

```bash
cd backend
python train/train.py
```

This updates:
- `backend/app/models/*.pkl`
- `backend/data/user_profiles.json`

Then restart the backend server.

## Key API Endpoints

- `POST /score-transaction` - score one transaction
- `GET /user-profile/{user_id}` - fetch profile used for feature context
- `GET /transactions/{user_id}` - list transaction history for user
- `GET /analyst-logs` - list all analyst logs
- `POST /review-log/{log_id}` - update review status

## Notes for Local Development

- Run backend on port 8000 and frontend on port 5173.
- CORS is configured in backend for local frontend URLs.
- If frontend reports `'vite' is not recognized`, run `npm install` inside `frontend/`.

## Demo Flow

1. Open frontend and fill transaction details.
2. Set phone location on the map.
3. Submit the transaction for scoring.
4. Review risk label, score breakdown, and reason list.
5. Open heatmap/transactions views to inspect history and activity zones.
