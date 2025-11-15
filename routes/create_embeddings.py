# routes/create_embeddings.py
import pathlib
import json
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import zipfile

router = APIRouter()

@router.post("/create-embeddings")
async def create_embeddings(file: UploadFile = File(...)):
    try:
        # === Save uploaded file ===
        upload_dir = pathlib.Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # === Read file content ===
        if file.filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text_content = json.dumps(data, indent=2)
        elif file.filename.endswith(".csv") or file.filename.endswith(".txt"):
            text_content = file_path.read_text(encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # === Split text into chunks ===
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", " ", ".", ","]
        )
        chunks = splitter.split_text(text_content)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # === Create FAISS index ===
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.from_documents(documents, embeddings)

        # === Save FAISS index locally ===
        index_dir = pathlib.Path("faiss_indexes") / file.filename
        index_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(index_dir))

        # === Zip the FAISS folder for download ===
        zip_path = pathlib.Path("faiss_indexes") / f"{file.filename}.zip"
        shutil.make_archive(base_name=str(zip_path).replace(".zip", ""), format="zip", root_dir=index_dir)

        return FileResponse(str(zip_path), media_type="application/zip", filename=f"{file.filename}.zip")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
