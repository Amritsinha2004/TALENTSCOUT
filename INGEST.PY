import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import zipfile
import io
import os
import hashlib
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THIS TO YOUR ZIP PATH
ZIP_FILE_PATH = r"C:\Users\asus\Downloads\archive (2).zip"

# PATHS (Must be relative to this file to match Backend)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "local_pdf_db")
PDF_STORAGE_PATH = os.path.join(BASE_DIR, "stored_pdfs") 
COLLECTION_NAME = "corporate_resumes" # <--- MUST MATCH BACKEND

def setup_db():
    print(f"⏳ Connecting to DB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def extract_text(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except:
        return ""

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i + 500]) for i in range(0, len(words), 450)]

def main():
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"❌ Zip file not found: {ZIP_FILE_PATH}")
        return

    if not os.path.exists(PDF_STORAGE_PATH):
        os.makedirs(PDF_STORAGE_PATH)

    collection = setup_db()
    
    print(f"📂 Processing {ZIP_FILE_PATH}...")
    success_count = 0

    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith('.pdf')][:200]
        
        for i, filename in enumerate(pdf_files):
            try:
                pdf_bytes = z.read(filename)
                text = extract_text(pdf_bytes)
                if not text.strip(): continue

                # Save Physical PDF for UI
                clean_name = os.path.basename(filename).replace(" ", "_")
                with open(os.path.join(PDF_STORAGE_PATH, clean_name), "wb") as f:
                    f.write(pdf_bytes)

                # Save Vectors
                chunks = chunk_text(text)
                ids = [hashlib.md5(f"{clean_name}_{idx}".encode()).hexdigest() for idx in range(len(chunks))]
                metadatas = [{"source": clean_name} for _ in chunks]

                collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                success_count += 1
                
                if success_count % 10 == 0: print(f"   Processed {success_count}...")
            except Exception as e:
                print(f"   Skipped {filename}: {e}")

    print(f"🎉 Done! Processed {success_count} resumes.")
    print("Now run 'python Backend.py'")

if __name__ == "__main__":
    main()
