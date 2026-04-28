import os
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

mcp = FastMCP("obama-advisor")
model = SentenceTransformer("all-MiniLM-L6-v2")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

@mcp.tool()
def search_obama_context(query: str) -> str:
    """
    Search Barack Obama's speeches for relevant context.
    ALWAYS call this before responding as Obama to ground your answer
    in his actual words and positions.
    """
    vec = model.encode(query).tolist()
    
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

if __name__ == "__main__":
    mcp.run(transport="streamable-http")