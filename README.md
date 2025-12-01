# SnapTrack

OCR Receipt Budgeting tool with a FastAPI backend and React + Vite + Tailwind frontend.

## Project layout
```
backend/                 # FastAPI service
  main.py                # API routes and end-to-end OCR pipeline
  cropReceipt.py         # Receipt auto-cropping / deskew logic
  extract_amount.py      # EasyOCR parsing helpers (totals, taxes, items)
  ocr_easy.py            # EasyOCR wrapper
  requirements.txt       # Python dependencies
frontend/                # React + Vite + Tailwind UI
  src/
    components/          # UI components (upload, preview, results, summary)
    api.js               # Axios wrapper targeting http://localhost:5050
    App.jsx              # UI composition and state management
    index.css            # Tailwind setup + neon theme utilities
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
- `POST /save_receipt` (JSON) → Persist receipt metadata and image to disk
- `GET /history` → Saved receipts and monthly totals

## Processing pipeline
- **Receipt cropping**: `backend/cropReceipt.py` (`detect_receipt_lines`).
- **OCR extraction**: EasyOCR via `backend/ocr_easy.py`.
- **Parsing**: Totals, subtotal, taxes, store name, and line items via `backend/extract_amount.py`.
- **Persistence**: `/save_receipt` stores JSON metadata and the receipt image under `backend/data`.
- **History**: `/history` aggregates saved receipts for monthly budgeting.

Upload an image from the frontend, click **Scan Receipt**, and watch the parsed items and totals render in the neon-themed dashboard.
