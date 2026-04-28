import os
import requests as http_requests
from mcp.server.fastmcp import FastMCP
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

mcp = FastMCP("obama-advisor")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def embed(text: str) -> list:
    response = http_requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Content-Type": "application/json"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()

@mcp.tool()
def search_obama_context(query: str) -> str:
    """
    Search Barack Obama's speeches for relevant context.
    ALWAYS call this before responding as Obama to ground your answer
    in his actual words and positions.
    """
    vec = embed(query)
    res = supabase.rpc("match_obama_speeches", {
        "query_embedding": vec,
        "match_count": 5
    }).execute()

    if not res.data:
        return "No relevant context found."

    passages = []
    for row in res.data:
        passages.append(f"[{row['title']}, {row['date']}]\n{row['chunk']}")

    return (
        "CONTEXT FROM OBAMA'S ACTUAL SPEECHES:\n\n"
        + "\n\n---\n\n".join(passages)
        + "\n\nNow respond AS Obama, in first person, in his voice."
    )

app = mcp.sse_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)