from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pathlib
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import json

router = APIRouter()

class Question(BaseModel):
    question: str

@router.post("/chat/{chatid}")
async def chat(chatid: str, payload: Question):
    try:
        BASE_DIR = pathlib.Path(__file__).resolve().parent
        index_dir = BASE_DIR / ".." / "chatbots" / chatid / "faiss_index"

        if not index_dir.exists():
            raise HTTPException(status_code=404, detail=f"FAISS index not found: {index_dir}")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        # Load instructions from JSON
        instructions_path = BASE_DIR / ".." / "chatbots" / chatid / "instructions.json"
        if not instructions_path.exists():
            raise HTTPException(status_code=404, detail="Instructions not found")
        with open(instructions_path, "r", encoding="utf-8") as f:
            instructions_data = json.load(f)
        instructions_text = instructions_data.get("instructions", "You are a helpful assistant.")


        docs = retriever.invoke(payload.question)
        context = "\n\n".join([d.page_content for d in docs])
        safe_context = context.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", instructions_text),
            ("system", "Context:\n{context}"),
            ("human", "{question}")
        ])
        final_prompt = prompt.format(context=safe_context, question=payload.question)


        # Call LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
        response = llm.invoke([HumanMessage(content=final_prompt)])

        return {"answer": response.content}

    except Exception as e:
        print("Error in /chat route:", e)
        raise HTTPException(status_code=500, detail=str(e))
