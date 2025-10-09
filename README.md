# 📊 DataNarrator

**DataNarrator** is a full-stack AI-powered data analysis tool that transforms any CSV file into an executive narrative report.  
It combines **FastAPI (Python)** for statistical intelligence with **Next.js (React)** for an intuitive user interface and **Ollama LLM** for automated report generation.

---

## 🧠 Overview

- Upload CSV datasets directly in the browser.  
- The backend performs numerical and statistical analysis (trends, regressions, correlations).  
- An AI model generates a written executive summary based on the data.  
- Results can be exported to a styled **PDF report**.  
- The entire system is containerized via **Docker Compose** for portability.

---

## ⚙️ Tech Stack

| Layer | Technology | Description |
|-------|-------------|-------------|
| Frontend | Next.js, React, TailwindCSS | Modern and responsive UI |
| Backend | FastAPI, Python | Statistical computation and API |
| AI Layer | Ollama (Llama3) | Natural language report generation |
| Data | Pandas, NumPy, SciPy | Trend detection and correlation analysis |
| Infrastructure | Docker Compose | Multi-service orchestration |
| Export | jsPDF | Executive-style PDF generation |

---

## 📁 Project Structure

```
data-narrator/
│
├── backend/
│   ├── main.py               # FastAPI backend and analysis logic
│   ├── requirements.txt      # Python dependencies
│
├── frontend/
│   ├── index.jsx             # Main React page (upload, analysis, export)
│   ├── package.json          # Frontend dependencies
│
├── docker-compose.yml        # Service orchestration
├── README.md                 # Project documentation
└── .env.example              # Optional environment variables
```

---

## 🚀 How to Run

### Option 1 — Using Docker (recommended)
```bash
git clone https://github.com/EnricoMaragno/data-narrator.git
cd data-narrator
docker compose up --build
```

The application will be available at:
- **Frontend:** http://localhost:3000  
- **Backend (API):** http://127.0.0.1:8000  

---

### Option 2 — Run manually (development mode)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 📄 Output Example

The exported PDF includes:
- Dataset summary (rows, columns, numeric fields)
- Statistically verified trends and correlations
- Actionable insights and executive recommendations
- Timestamp and author signature

---

## 👤 Author

**Enrico Maragno**  
Barcelona, Spain 🇪🇸  
Full Stack & Data Engineer – passionate about building AI-powered analytical products.

---

## 🧱 License

MIT License © 2025 Enrico Maragno
