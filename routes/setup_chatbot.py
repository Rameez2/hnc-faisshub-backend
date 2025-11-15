import os
import uuid
import json
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables for local testing
load_dotenv()

# --- Configuration (MUST BE SET IN YOUR DEPLOYMENT ENVIRONMENT) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "fias") # Your Supabase bucket name


# --- API Setup ---
router = APIRouter()
STORAGE_API_URL = f"{SUPABASE_URL}/storage/v1"

# --- Utility Functions (Kept as is) ---

async def upload_file_bytes(file_path: str, content_type: str, content: bytes):
    """Handles the direct upload of file bytes to the Supabase Storage API."""
    upload_url = f"{STORAGE_API_URL}/object/{BUCKET_NAME}/{file_path}"
    
    upload_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": content_type
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            upload_url,
            content=content,
            headers=upload_headers
        )

        if response.status_code != 200:
            print(f"Supabase Upload Error ({response.status_code}): {response.text}")
            try:
                error_detail = response.json().get('error', response.text)
            except json.JSONDecodeError:
                error_detail = response.text
                
            raise HTTPException(
                status_code=500,
                detail=f"Storage upload failed for {file_path}: {error_detail}"
            )
        
        # Returns the full public URL path
        return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"


# --- REVISED API Endpoint ---

@router.post("/setup-chatbot")
async def setup_chatbot(
    faiss_file: UploadFile = File(..., description="The FAISS index file (.faiss)"),
    pkl_file: UploadFile = File(..., description="The metadata/index file (.pkl)"),
    # instructions: str = Form(...) # Uncomment if you want to use this
):
    """
    Receives the .faiss and .pkl index files and uploads them 
    to a unique folder in Supabase Storage.
    """
    
    # 1. Generate unique ID for the storage folder
    chatbot_id = str(uuid.uuid4())[:12] # Use 12 chars for better uniqueness
    folder_path = f"{chatbot_id}" 

    uploaded_files_info = {}

    try:
        # 2. Validation
        if not faiss_file.filename.endswith(".faiss"):
            raise HTTPException(status_code=400, detail="The FAISS file must have a '.faiss' extension.")
        if not pkl_file.filename.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="The PKL file must have a '.pkl' extension.")
        
        # --- 3. Upload .faiss File ---
        faiss_bytes = await faiss_file.read()
        faiss_storage_path = f"{folder_path}/{faiss_file.filename}"
        print(f"Uploading .faiss to: {faiss_storage_path}")
        
        faiss_url = await upload_file_bytes(
            file_path=faiss_storage_path, 
            content_type="application/octet-stream", # Standard binary type
            content=faiss_bytes
        )
        uploaded_files_info["faiss_url"] = faiss_url


        # --- 4. Upload .pkl File ---
        pkl_bytes = await pkl_file.read()
        pkl_storage_path = f"{folder_path}/{pkl_file.filename}"
        print(f"Uploading .pkl to: {pkl_storage_path}")

        pkl_url = await upload_file_bytes(
            file_path=pkl_storage_path, 
            content_type="application/octet-stream", # Standard binary type
            content=pkl_bytes
        )
        uploaded_files_info["pkl_url"] = pkl_url


        # 5. Success Response
        return JSONResponse(content={
            "message": "✅ Chatbot setup successfully! FAISS and PKL files stored in Supabase.", 
            "chatbot_id": chatbot_id,
            **uploaded_files_info,
        })

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        # Catch all other exceptions
        print(f"Unhandled setup error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred during setup: {str(e)}"
        )