import os
import shutil
import hashlib
import zipfile
import logging
import re
import sqlite3
import sys
import random 
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from passlib.context import CryptContext
from jose import jwt, JWTError
from fpdf import FPDF
import ollama  # <--- OFFICIAL AI LIBRARY

# --- 1. CONFIGURATION ---
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent
    CHROMA_PATH: Path = BASE_DIR / "local_pdf_db"
    STORAGE_PATH: Path = BASE_DIR / "stored_pdfs"
    TEMP_PATH: Path = BASE_DIR / "temp_uploads"
    USER_DB_PATH: Path = BASE_DIR / "users.db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    COLLECTION_NAME: str = "corporate_resumes" 
    
    # SECURITY CONFIG
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY" 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

settings = Settings()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResumeEngine")

# Ensure folders exist
for path in [settings.STORAGE_PATH, settings.TEMP_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# --- 2. SECURITY & DATABASE SETUP ---
try:
    pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
except Exception:
    logger.critical("Install argon2-cffi: pip install argon2-cffi")
    sys.exit(1)

def init_db():
    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT, full_name TEXT)''')
    try:
        c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
    except:
        pass
    conn.commit()
    conn.close()

init_db()

# In-Memory OTP Storage
otp_storage = {} 

class AuthHandler:
    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)
    @staticmethod
    def create_access_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

# --- 3. DATA MODELS ---
class UserSignup(BaseModel):
    username: str
    password: str
    full_name: str
    role: str

class UserLogin(BaseModel):
    username: str
    password: str

class OTPVerify(BaseModel):
    username: str
    otp: str

class ResumeFormData(BaseModel):
    full_name: str
    email: str
    job_title: str
    summary: str
    skills: str
    experience: str
    education: str

class ChatRequest(BaseModel):
    filename: str
    question: str

# --- 4. CORE LOGIC: VECTOR DB & AI ---
class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME, 
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, filename: str, chunks: List[str]):
        if not chunks: return
        ids = [hashlib.md5(f"{filename}_{i}".encode()).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    def generate_ai_summary(self, context_text, query):
        """
        Uses Llama 3.2 to convert raw text chunks into a smart summary.
        """
        prompt = f"""
        Context from Resume: "{context_text}"
        User Search Query: "{query}"
        
        Instruction: The user is searching for a candidate. Based ONLY on the context provided, write a one-sentence summary highlighting the candidate's experience relevant to the search query. If the context is irrelevant, just return the context text.
        """
        
        try:
            response = ollama.generate(model='llama3.2', prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            print(f"Ollama AI Error: {e}")
            return context_text[:200] + "..."

    def search(self, query: str, limit: int = 5):
        results = self.collection.query(
            query_texts=[query], n_results=limit, include=['documents', 'metadatas', 'distances']
        )
        structured = []
        if results['documents']:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                score = round((1 - dist) * 100, 1)
                
                # Only run AI Summary if match is good (>30%) to keep it fast
                if score > 30:
                    ai_summary = self.generate_ai_summary(doc, query)
                else:
                    ai_summary = doc[:200] + "..." # Raw text fallback

                structured.append({"text": ai_summary, "source": meta['source'], "score": score})
        return structured

class PDFGenerator:
    @staticmethod
    def create_resume_pdf(data: ResumeFormData, output_path: Path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 10, data.full_name, ln=True)
        pdf.set_font("Arial", "I", 14)
        pdf.cell(0, 10, data.job_title, ln=True)
        pdf.ln(10)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "CONTACT", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, data.email, ln=True)
        pdf.ln(5)

        sections = [("SUMMARY", data.summary), ("SKILLS", data.skills), 
                    ("EXPERIENCE", data.experience), ("EDUCATION", data.education)]
        
        for title, content in sections:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 8, content)
            pdf.ln(5)
            
        pdf.output(str(output_path))

# --- 5. APP INITIALIZATION ---
db = VectorDB()
app = FastAPI(title="TalentScout API")

app.mount("/pdfs", StaticFiles(directory=str(settings.STORAGE_PATH)), name="pdfs")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. PAGE ROUTES ---
@app.get("/")
def serve_landing():
    return FileResponse(settings.BASE_DIR / "index.html")

@app.get("/{page_name}.html")
def serve_pages(page_name: str):
    file_path = settings.BASE_DIR / f"{page_name}.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(file_path)

# --- 7. AUTH API ROUTES ---
@app.post("/auth/signup")
def signup(user: UserSignup):
    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    try:
        hashed_pw = AuthHandler.get_password_hash(user.password)
        c.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                  (user.username, hashed_pw, user.role, user.full_name))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"message": "Account created"}

@app.post("/auth/login")
def login(creds: UserLogin):
    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username=?", (creds.username,))
    result = c.fetchone()
    conn.close()

    if not result or not AuthHandler.verify_password(creds.password, result[0]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # --- UNIVERSAL OTP ---
    otp_code = str(random.randint(100000, 999999))
    otp_storage[creds.username] = otp_code
    
    # PRINT OTP TO TERMINAL
    print(f"\n" + "="*40)
    print(f"🔐 SECURITY ALERT: OTP for User '{creds.username}'")
    print(f"👉 CODE: {otp_code}")
    print(f"="*40 + "\n")

    return {"message": "OTP_REQUIRED", "username": creds.username}

@app.post("/auth/verify-otp")
def verify_otp(data: OTPVerify):
    if data.username not in otp_storage:
        raise HTTPException(400, "OTP expired or not generated")
    
    stored_code = otp_storage[data.username]
    if data.otp != stored_code:
        raise HTTPException(401, "Invalid OTP Code")

    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    c.execute("SELECT role, full_name FROM users WHERE username=?", (data.username,))
    user_data = c.fetchone()
    conn.close()

    del otp_storage[data.username]

    token = AuthHandler.create_access_token({"sub": data.username, "role": user_data[0]})
    return {"access_token": token, "token_type": "bearer", "role": user_data[0], "name": user_data[1]}

# --- 8. BUSINESS LOGIC ROUTES ---
@app.get("/search")
def search_resumes(query: str):
    results = db.search(query)
    return {"results": results, "count": len(results)}

@app.post("/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    for file in files:
        safe_name = os.path.basename(file.filename).replace(" ", "_")
        dest = settings.STORAGE_PATH / safe_name
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            reader = PdfReader(dest)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = [text[i:i+500] for i in range(0, len(text), 450)]
            db.add_document(safe_name, chunks)
        except Exception as e:
            print(f"Error parsing {safe_name}: {e}")
            
    return {"message": "Upload complete"}

@app.post("/create-resume")
async def create_resume(data: ResumeFormData):
    safe_name = f"{data.full_name.replace(' ', '_')}_Resume.pdf"
    dest = settings.STORAGE_PATH / safe_name
    
    try:
        PDFGenerator.create_resume_pdf(data, dest)
        full_text = f"{data.full_name} {data.job_title} {data.skills} {data.summary} {data.experience}"
        db.add_document(safe_name, [full_text])
        return {"message": "Resume created", "filename": safe_name}
    except Exception as e:
        raise HTTPException(500, str(e))

# --- 9. NEW CHAT ENDPOINT (ADDED THIS BACK IN) ---
@app.post("/chat")
def chat_with_resume(data: ChatRequest):
    # 1. Look for the file in the database
    results = db.collection.get(
        where={"source": data.filename},
        include=["documents"]
    )
    
    # 2. If file not found, return error
    if not results['documents']:
        raise HTTPException(404, "Resume data not found. Try re-uploading the PDF.")

    # 3. Prepare text for AI
    full_text = "\n".join(results['documents'])[:3000] 

    # 4. Ask Ollama (AI Engine)
    try:
        prompt = f"Resume: {full_text}\nQuestion: {data.question}\nAnswer:"
        response = ollama.generate(model='llama3.2', prompt=prompt)
        return {"answer": response['response'].strip()}
    except Exception as e:
        return {"answer": f"AI Error: {str(e)}. Is Ollama running?"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend:app", host="0.0.0.0", port=8080, reload=True)