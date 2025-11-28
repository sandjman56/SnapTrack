# SnapTrack

OCR Receipt Budgeting tool with a FastAPI backend and React + Vite + Tailwind frontend.

## Project layout
```
backend/                 # FastAPI service
  main.py                # API routes and end-to-end OCR pipeline
  cropReceipt.py         # Receipt auto-cropping / deskew logic
  findcontours.py        # Text contour preprocessing and detection
  model_loader.py        # CNN definition, weight loading, and prediction helper
  requirements.txt       # Python dependencies
frontend/                # React + Vite + Tailwind UI
  src/
    components/          # UI components (upload, preview, results, summary)
    api.js               # Axios wrapper targeting http://localhost:5050
    App.jsx              # UI composition and state management
    index.css            # Tailwind setup + neon theme utilities
machine_learning_model_victor/  # Provided CNN weights and training script
```

## Running locally
### Backend (http://localhost:5050)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 5050
```

### Frontend (http://localhost:5173)
```bash
cd frontend
npm install
npm run dev
```

## API
- `GET /health` → `{ "status": "ok" }`
- `POST /process_receipt` (multipart/form-data, field `file`) → Parsed receipt JSON

## Processing pipeline
- **Receipt cropping**: `backend/cropReceipt.py` (`detect_receipt_lines`).
- **Contour detection**: `backend/findcontours.py` (`preprocess`, `find_text_contours`).
- **ML OCR inference**: `backend/model_loader.py` (`predict_character`).
- **Text reconstruction**: `reconstruct_text` in `backend/main.py`.
- **Parsing into items**: `parse_receipt_text` in `backend/main.py`.
- **Returning JSON**: `POST /process_receipt` in `backend/main.py`.

Upload an image from the frontend, click **Scan Receipt**, and watch the parsed items and totals render in the neon-themed dashboard.
