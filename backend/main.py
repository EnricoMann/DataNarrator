# main.py — DataNarrator API (v4.7 - Accessible PDF via /files)
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import io, os, logging
from typing import Optional, List, Dict
from datetime import datetime
import ollama
from scipy.stats import linregress, pearsonr
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
app = FastAPI(title="DataNarrator API", version="4.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataNarrator")

# Mount /files for serving generated PDFs
REPORT_DIR = "/mnt/data"
os.makedirs(REPORT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=REPORT_DIR), name="files")

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def load_to_df(contents: bytes, filename: str) -> pd.DataFrame:
    ext = (filename or "").split(".")[-1].lower()
    bio = io.BytesIO(contents)
    try:
        if ext == "csv":
            try:
                return pd.read_csv(bio)
            except Exception:
                bio.seek(0)
                return pd.read_csv(bio, sep=";", encoding="utf-8")
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(bio)
        elif ext == "json":
            return pd.read_json(bio)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"Failed to parse file: {e}")

def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    hints = ("date", "day", "time", "month", "year", "period")
    for col in df.columns:
        if any(h in col.lower() for h in hints):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
    return None

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.copy()
    for col in num_df.columns:
        if pd.api.types.is_numeric_dtype(num_df[col]):
            continue
        cleaned = (
            num_df[col]
            .astype(str)
            .str.replace(r"[^\d\-\.,]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().sum() > len(df) * 0.3:
            num_df[col] = coerced
    return num_df

def significance_label(p: float) -> str:
    if p < 0.05:
        return "verified"
    elif p < 0.10:
        return "marginal"
    return "uncertain"

# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def analyze_dataset(df: pd.DataFrame, time_col: Optional[str], numeric_cols: List[str]):
    def trend_strength(r2, p):
        if p >= 0.05:
            return "insignificant"
        if r2 > 0.8:
            return "strong"
        if r2 > 0.5:
            return "moderate"
        if r2 > 0.3:
            return "weak"
        return "insignificant"

    trends, corrs, regs = {}, [], []

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        for col in numeric_cols:
            y = df[col].astype(float)
            if y.notna().sum() < 3:
                continue
            yy = y[y.notna()].to_numpy()
            tt = np.arange(len(yy))
            slope, intercept, r, p, stderr = linregress(tt, yy)
            r2 = float(r ** 2)
            trends[col] = {
                "slope": round(slope, 6),
                "r2": round(r2, 6),
                "p_value": round(p, 6),
                "significance": significance_label(p),
                "trend_strength": trend_strength(r2, p),
            }

    num_df = df.select_dtypes(include=[np.number])
    cols = num_df.columns.tolist()
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            sa, sb = num_df[a], num_df[b]
            mask = sa.notna() & sb.notna()
            if mask.sum() < 3:
                continue
            r, p = pearsonr(sa[mask], sb[mask])
            corrs.append({
                "x": a, "y": b,
                "r": round(r, 3), "p_value": round(p, 6),
                "significance": significance_label(p)
            })

    for c in corrs[:5]:
        x, y = c["x"], c["y"]
        x_vals, y_vals = df[x].astype(float), df[y].astype(float)
        mask = x_vals.notna() & y_vals.notna()
        if mask.sum() < 3:
            continue
        slope, intercept, r, p, stderr = linregress(x_vals[mask], y_vals[mask])
        regs.append({
            "x": x, "y": y,
            "slope": round(slope, 6),
            "r2": round(r ** 2, 6),
            "p_value": round(p, 6),
            "significance": significance_label(p)
        })
    return trends, corrs, regs

# ----------------------------------------------------------------------
# PDF Generator
# ----------------------------------------------------------------------
def create_pdf(summary_text: str, dataset_info: Dict, out_path: str):
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("DataNarrator Executive Report", styles["Title"]))
    elems.append(Spacer(1, 0.3 * cm))

    meta = f"""
    <b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}<br/>
    <b>File:</b> {dataset_info.get('filename', 'N/A')}<br/>
    <b>Rows:</b> {dataset_info.get('rows', 'N/A')} • <b>Columns:</b> {dataset_info.get('cols', 'N/A')}<br/>
    <b>Detected Time Column:</b> {dataset_info.get('time_column', 'None')}
    """
    elems.append(Paragraph(meta, styles["Normal"]))
    elems.append(Spacer(1, 0.5 * cm))

    elems.append(Paragraph("<b>Executive Analysis</b>", styles["Heading2"]))
    for para in summary_text.split("\n"):
        if para.strip():
            elems.append(Paragraph(para.strip(), styles["BodyText"]))
            elems.append(Spacer(1, 0.2 * cm))

    doc.build(elems)
    return out_path

# ----------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    export_pdf_report: bool = Query(False, description="Generate PDF after analysis"),
):
    try:
        contents = await file.read()
        df_raw = load_to_df(contents, file.filename)
        if df_raw.empty:
            raise ValueError("Dataset is empty.")

        df = coerce_numeric(df_raw)
        time_col = detect_datetime_column(df)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        trends, corrs, regs = analyze_dataset(df, time_col, numeric_cols)

        n_rows, n_cols = len(df), len(df.columns)

        trends_text = "\n".join([f"{k}: slope={v['slope']}, R²={v['r2']}, p={v['p_value']}, significance={v['significance']}" for k, v in trends.items()])
        corr_text = "\n".join([f"{d['x']} ↔ {d['y']}: r={d['r']}, p={d['p_value']}, significance={d['significance']}" for d in corrs[:5]])

        prompt = f"""
You are a senior data analyst writing a concise executive report.
Dataset: {n_rows} rows, {n_cols} columns ({file.filename})
Detected time column: {time_col or "None"}

[Trends]
{trends_text}

[Correlations]
{corr_text}

Provide a professional executive summary including:
- verified and marginal trends
- concise insights
- actionable business recommendations
- limitations and next steps
        """

        model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        summary = response["message"]["content"].strip()

        pdf_url = None
        if export_pdf_report:
            pdf_path = os.path.join(REPORT_DIR, "executive_analysis.pdf")
            create_pdf(summary, {"filename": file.filename, "rows": n_rows, "cols": n_cols, "time_column": time_col}, pdf_path)
            pdf_url = f"http://127.0.0.1:8000/files/executive_analysis.pdf"

        return {
            "summary": summary,
            "pdf_url": pdf_url,
            "metadata": {
                "filename": file.filename,
                "rows": n_rows,
                "cols": n_cols,
                "time_column": time_col,
                "numeric_columns": numeric_cols,
                "trends": trends,
                "correlations": corrs[:5],
                "regressions": regs[:5],
            },
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis process failed.")
