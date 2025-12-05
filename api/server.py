import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mangum import Mangum  # ✅ REQUIRED FOR VERCEL

load_dotenv()

# === Load OpenAI key from ENV ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="RAGBuilder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "✅ FastAPI server is running!"}

# Import routes
from routes.create_embeddings import router as create_embeddings_router
from routes.setup_chatbot import router as setup_chatbot_router
from routes.chat import router as chat_router
from routes.ichat import router as i_chat_router

app.include_router(create_embeddings_router)
app.include_router(setup_chatbot_router)
app.include_router(chat_router)
app.include_router(i_chat_router)

# ✅ THIS LINE IS MANDATORY FOR VERCEL
handler = Mangum(app)
