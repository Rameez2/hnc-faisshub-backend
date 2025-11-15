import os
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pathlib, json, os, time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from supabase import create_client, Client
import httpx # Import httpx for connection handling and downloading
import tempfile # For creating temporary directories on the server
import shutil # For safely handling directory removals

router = APIRouter()


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Ensure you use a service_role key for this server code
BUCKET_NAME = os.environ.get("SUPABASE_BUCKET_NAME", "fias")

# -------------------------------
# 🌟 CACHING IMPLEMENTATION
# Cache structure: { "chatid": {"db": FAISS_object, "instructions": "string", "temp_dir": tempfile.TemporaryDirectory_object} }
# NOTE: The temp_dir must persist while the FAISS object is in memory.
FAISS_CACHE = {} 
# -------------------------------

# --- Configuration: Use Environment Variables for Security ---
BUCKET_NAME = os.environ.get("SUPABASE_BUCKET_NAME", "fias") # Default used if not set

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase configuration. Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are set.")

# Global Supabase client instance
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client():
    """Returns the global Supabase client, re-initializing if necessary."""
    global supabase
    return supabase

def re_initialize_supabase_client():
    """Re-initializes the global Supabase client. Used for connection issues."""
    global supabase
    print("DEBUG: Attempting to re-initialize Supabase client...")
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("DEBUG: Supabase client re-initialized successfully.")
        return supabase
    except Exception as e:
        print(f"DEBUG: Failed to re-initialize Supabase client: {e}")
        raise

class Question(BaseModel):
    question: str

# -------------------------------
# Helper: Download and Load FAISS Index 
# -------------------------------

async def download_file_from_storage(file_url: str, local_path: pathlib.Path):
    """Downloads a single file from Supabase Storage to a local path."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        print(f"DEBUG: Downloading file from: {file_url}")
        response = await client.get(file_url)

        if response.status_code == 404:
            print(f"DEBUG: 404 Error - File not found at URL: {file_url}")
            raise HTTPException(status_code=404, detail=f"Index file not found in storage at URL: {file_url}")
        if response.status_code != 200:
            print(f"DEBUG: Storage Download Error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=500, detail=f"Failed to download index file: {response.status_code}")
        
        # Write content to the local file path
        local_path.write_bytes(response.content)
        print(f"DEBUG: Successfully downloaded file to: {local_path}")

async def download_and_load_faiss_index(faiss_url: str, pkl_url: str, embeddings: OpenAIEmbeddings):
    """
    Downloads the individual .faiss and .pkl files, saves them to a temporary 
    directory, and loads the FAISS object.
    
    Returns: db (FAISS object), temp_dir (tempfile.TemporaryDirectory object)
    """
    
    FAISS_FILE = "index.faiss" 
    PKL_FILE = "index.pkl" 
    
    # tempfile.TemporaryDirectory is used to manage the lifetime of the directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = pathlib.Path(temp_dir.name)
    print(f"DEBUG: Created temporary directory: {temp_path}")

    faiss_local_path = temp_path / FAISS_FILE
    pkl_local_path = temp_path / PKL_FILE
    
    # We intentionally do not use a try/except here so the caller handles cleanup on failure
    print("DEBUG: Attempting to download FAISS index files using database URLs...")

    # 2. Download the .faiss file using the provided URL
    await download_file_from_storage(faiss_url, faiss_local_path)
    
    # 3. Download the .pkl file using the provided URL
    await download_file_from_storage(pkl_url, pkl_local_path)
    
    # 4. Load the FAISS index from the temporary directory
    print(f"DEBUG: Loading FAISS index from local path: {temp_path}")
    db = FAISS.load_local(str(temp_path), embeddings, allow_dangerous_deserialization=True)
    print("DEBUG: FAISS index loaded successfully.")
    
    return db, temp_dir


# -------------------------------
# POST: Handle chatbot messages
# -------------------------------
@router.post("/ichat/{chatid}")
async def chat(chatid: str, payload: Question, x_api_key: str = Header(...)):
    global FAISS_CACHE # Declare access to the global cache
    MAX_RETRIES = 2
    
    # These variables hold the final data/objects required for the RAG step.
    db = None
    instructions_text = None
    # temp_dir is ONLY used for non-cached attempts.
    temp_dir_to_cleanup = None 
    
    # Variables retrieved from DB on cache miss
    faiss_url = None 
    pkl_url = None
    
    print(f"\n--- DEBUG START: Request for chatid={chatid} ---")

    # --- 1️⃣ CACHE CHECK BEFORE DB OPERATION ---
    if chatid in FAISS_CACHE:
        print(f"DEBUG: Cache hit for chatid={chatid}. Using cached FAISS index.")
        cached_data = FAISS_CACHE[chatid]
        db = cached_data["db"]
        instructions_text = cached_data["instructions"]
        # Skip the rest of the DB/Download section
    else:
        # --- 🔴 CACHE MISS: Perform DB Lookup and Download ---
        
        # This loop handles DB fetching and connection retries
        for attempt in range(MAX_RETRIES):
            try:
                db_client = get_supabase_client()
                
                # 1️⃣ Validate API key (Using limit(1) for robustness)
                print(f"DEBUG: Attempt {attempt + 1}: Validating API key...")
                key_result = (
                    db_client.table("api_keys")
                    .select("*")
                    .eq("key_value", x_api_key)
                    .eq("active", True)
                    .limit(1) 
                    .execute()
                )

                if not key_result.data:
                    print("DEBUG: API Key validation FAILED. Raising 403.")
                    raise HTTPException(status_code=403, detail="Invalid or inactive API key")
                
                print("DEBUG: API Key validated successfully.")
                
                # 1b) Update last_used_at (Fix for 204 error)
                print("DEBUG: Attempting to update last_used_at...")
                # db_client.table("api_keys").update({"last_used_at": "now()"}).eq("key_value", x_api_key).execute(
                #     retrieve_single_row=False
                # )
                print("DEBUG: last_used_at updated (or update skipped gracefully).")
                
                # 2️⃣ Fetch instructions AND FAISS/PKL URLs from chatbots table (Using limit(1) for robustness)
                print(f"DEBUG: Fetching chatbot config for ID: {chatid}...")
                chatbot_result = (
                    db_client.table("chatbots")
                    .select("instructions, faiss_url, pkl_url") 
                    .eq("chatbot_id", chatid)
                    .limit(1)
                    .execute()
                )
                
                if not chatbot_result.data:
                    print(f"DEBUG: Chatbot config for ID {chatid} not found. Raising 404.")
                    raise HTTPException(status_code=404, detail="Chatbot not found")

                data = chatbot_result.data[0] # Get the single row
                instructions_text = data.get("instructions", "You are a helpful assistant.")
                faiss_url = data.get("faiss_url")
                pkl_url = data.get("pkl_url")
                
                print(f"DEBUG: Instructions fetched: {instructions_text[:50]}...")
                print(f"DEBUG: FAISS URL: {faiss_url[:60]}...")
                print(f"DEBUG: PKL URL: {pkl_url[:60]}...")


                if not faiss_url or not pkl_url:
                    print("DEBUG: Missing FAISS/PKL URL in DB. Raising 500.")
                    raise HTTPException(status_code=500, detail="Chatbot configuration error: Missing FAISS or PKL URL in database.")

                # If all Supabase operations succeeded, break the retry loop
                print("DEBUG: Database configuration fetched successfully. Breaking retry loop.")
                break 
            
            except httpx.RemoteProtocolError as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"DEBUG: Supabase connection failed (Attempt {attempt + 1}). Retrying in 1 second...")
                    re_initialize_supabase_client()
                    time.sleep(1)
                    continue
                else:
                    print(f"DEBUG: Supabase connection failed after {MAX_RETRIES} attempts. Raising 503.")
                    raise HTTPException(status_code=503, detail=f"Database connection failed after multiple retries. Error: {str(e)}")
            
            except Exception as e:
                # Handle non-connection related errors (like 403, 404 from above) immediately
                print(f"DEBUG: Non-retryable error in DB phase: {type(e).__name__}: {str(e)}")
                raise e
        
        # --- 3️⃣ Download and Load (Only on Cache Miss) ---
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Pass the retrieved URLs to the helper function
            print("DEBUG: Starting FAISS download and load process.")
            db, temp_dir = await download_and_load_faiss_index(faiss_url, pkl_url, embeddings) 
            
            # 4. Store the loaded objects and its temporary directory in the global cache
            FAISS_CACHE[chatid] = {
                "db": db, 
                "instructions": instructions_text,
                "temp_dir": temp_dir # Keep this reference!
            }
            print(f"DEBUG: FAISS index for chatid={chatid} cached successfully.")

        except Exception as e:
            # If download/load fails, ensure the temporary directory is cleaned up
            if 'temp_dir' in locals() and temp_dir is not None:
                 temp_dir.cleanup()
            print(f"DEBUG: Error during FAISS loading on cache miss: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading vector index: {str(e)}")


    # --- 4️⃣ RAG Execution (Common path for Cache Hit and Cache Miss) ---
    try:
        # Check if DB was successfully set (either from cache or by download/load)
        if db is None or instructions_text is None:
            # This should not be reachable if all prior checks were correct
             raise HTTPException(status_code=500, detail="Internal configuration error: Index object is missing.")
             
        print("DEBUG: Starting RAG retrieval.")
        retriever = db.as_retriever(search_kwargs={"k": 4})

        docs = retriever.invoke(payload.question)
        context = "\n\n".join([d.page_content for d in docs])
        print(f"DEBUG: Retrieved {len(docs)} documents. Context length: {len(context)} chars.")
        
        # LLM Call
        print("DEBUG: Calling LLM (gpt-4o-mini).")
        safe_context = context.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", instructions_text),
            ("system", "Context:\n{context}"),
            ("human", "{question}")
        ])
        final_prompt = prompt.format(context=safe_context, question=payload.question)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
        response = llm.invoke([HumanMessage(content=final_prompt)])
        print("DEBUG: LLM response received successfully.")
        return {"answer": response.content}

    except HTTPException as e:
        print(f"DEBUG: HTTPException in RAG/LLM phase: {str(e.detail)}")
        raise e
    except Exception as e:
        import traceback
        print("DEBUG: General error in RAG/LLM phase. Printing traceback.")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # The cleanup responsibility for the cached temp_dir is now delegated to the process
        # shutdown. This finally block is no longer needed to clean up on success 
        # for a cache miss, as the directory is kept for the cache.
        print("--- DEBUG END ---")


# -------------------------------
# GET: Embed HTML iframe (No change needed)
# -------------------------------
@router.get("/ichat/{chatid}", response_class=HTMLResponse)
async def embed_chatbot(chatid: str):
# ... (HTML response code remains unchanged) ...
    """
    Returns a simple HTML iframe UI.
    The iframe JS will call POST /ichat/{chatid} with x-api-key header.
    """
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chatbot Embed</title>
<style>
  body {{ margin:0; font-family:'Inter',sans-serif; }}
  .chat-container {{ width:100%; height:100%; display:flex; flex-direction:column; border:2px solid #4f46e5; border-radius:12px; overflow:hidden; background:#f3f4f6; }}
  .chat-header {{ background:#4f46e5; color:white; padding:12px; font-weight:bold; display:flex; align-items:center; gap:8px; }}
  .chat-messages {{ flex:1; padding:12px; overflow-y:auto; }}
  .message {{ padding:8px 12px; border-radius:12px; margin-bottom:8px; max-width:80%; word-wrap:break-word; }}
  .user-msg {{ background:#4f46e5; color:white; align-self:flex-end; margin-left: auto; }}
  .bot-msg {{ background:#e5e7eb; color:#111827; align-self:flex-start; margin-right: auto; }}
  .typing {{ display:inline-block; width:12px; height:12px; margin-right:4px; background-color:#4f46e5; border-radius:50%; animation:blink 1.4s infinite both; }}
  @keyframes blink {{ 0%,80%,100%{{opacity:0;}} 40%{{opacity:1;}} }}
  .chat-input {{ display:flex; padding:8px; border-top:1px solid #d1d5db; gap:8px; background:white; }}
  .chat-input input {{ flex:1; padding:8px 12px; border-radius:12px; border:1px solid #d1d5db; outline:none; }}
  .chat-input button {{ padding:8px 12px; background:#4f46e5; color:white; border:none; border-radius:12px; cursor:pointer; transition: background-color 0.2s; }}
  .chat-input button:hover {{ background-color: #3730a3; }}
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">🤖 Chatbot ID: {chatid}</div>
  <div class="chat-messages" id="chatMessages">
    <div class="bot-msg">Hello! Ask me anything.</div>
  </div>
  <form class="chat-input" id="chatForm">
    <input type="text" id="chatInput" placeholder="Type your message..." autocomplete="off" />
    <button type="submit">Send</button>
  </form>
</div>

<script>
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');
const urlParams = new URLSearchParams(window.location.search);
// The API key is passed via the 'api_key' query parameter from the embedding source
const API_KEY = urlParams.get("api_key"); 

if (!API_KEY) {{
    addMessage("Error: 'api_key' parameter is missing. Please embed the iframe with the required key.", false);
    document.getElementById('chatInput').disabled = true;
    document.getElementById('chatForm').querySelector('button').disabled = true;
}}

function addMessage(content, isUser) {{
  const div = document.createElement('div');
  div.className = 'message ' + (isUser ? 'user-msg' : 'bot-msg');
  // Use innerHTML to handle potential markdown or newline formatting from the bot's answer
  div.innerHTML = content.replace(/\\n/g, '<br>');
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}}

function addTyping() {{
  const typingDiv = document.createElement('div');
  typingDiv.className = 'bot-msg typing-indicator';
  typingDiv.innerHTML = '<span class="typing"></span><span class="typing"></span><span class="typing"></span> Typing...';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return typingDiv;
}}

chatForm.addEventListener('submit', async (e) => {{
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg || !API_KEY) return;
  
  addMessage(msg, true);
  chatInput.value = '';
  chatInput.disabled = true;
  chatForm.querySelector('button').disabled = true;

  const typingDiv = addTyping();

  try {{
    const response = await fetch('/ichat/{chatid}', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
      }},
      body: JSON.stringify({{ question: msg }})
    }});

    const data = await response.json();
    
    // Check for API errors (e.g., 403, 404, 500)
    if (!response.ok) {{
        const errorMessage = data.detail || 'An unknown error occurred.';
        addMessage('[API Error] ' + errorMessage, false);
    }} else {{
        addMessage(data.answer, false);
    }}
    
  }} catch (err) {{
    addMessage('Network Error: Could not connect to the server.', false);
    console.error(err);
  }} finally {{
    typingDiv.remove();
    chatInput.disabled = false;
    chatForm.querySelector('button').disabled = false;
    chatInput.focus();
  }}
}});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)