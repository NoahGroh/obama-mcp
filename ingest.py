import os, requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

SPEECHES = [
    # Frühe Karriere
    ("https://www.presidency.ucsb.edu/documents/keynote-address-the-2004-democratic-national-convention",
     "DNC Keynote 2004", "2004-07-27"),
    ("https://www.presidency.ucsb.edu/documents/remarks-announcing-candidacy-for-president-springfield-illinois",
     "Announcing Candidacy", "2007-02-10"),

    # Wahlkampf
    ("https://www.presidency.ucsb.edu/documents/address-the-national-constitution-center-philadelphia-more-perfect-union",
     "A More Perfect Union", "2008-03-18"),

    # Präsidentschaft
    ("https://www.presidency.ucsb.edu/documents/inaugural-address-5",
     "First Inaugural Address", "2009-01-20"),
    ("https://www.presidency.ucsb.edu/documents/address-before-joint-session-the-congress-1",
     "Address to Congress 2009", "2009-02-24"),
    ("https://www.presidency.ucsb.edu/documents/remarks-cairo",
     "Cairo Speech", "2009-06-04"),
    ("https://www.presidency.ucsb.edu/documents/remarks-accepting-the-nobel-peace-prize-oslo",
     "Nobel Peace Prize Speech", "2009-12-10"),
    ("https://www.presidency.ucsb.edu/documents/inaugural-address-15",
     "Second Inaugural Address", "2013-01-21"),
    ("https://www.presidency.ucsb.edu/documents/remarks-election-victory-celebration-chicago-illinois",
     "Election Night Victory Speech 2012", "2012-11-07"),
    ("https://www.presidency.ucsb.edu/documents/address-before-joint-session-the-congress-the-state-the-union-20",
     "State of the Union 2015", "2015-01-20"),
    ("https://www.presidency.ucsb.edu/documents/eulogy-the-funeral-service-for-pastor-clementa-c-pinckney-the-emanuel-african-methodist",
     "Charleston Eulogy", "2015-06-26"),
    ("https://www.presidency.ucsb.edu/documents/farewell-address-the-nation-from-chicago-illinois",
     "Farewell Address", "2017-01-10"),
]

def scrape(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    # Navigations-Elemente entfernen
    for tag in soup.find_all(["nav", "footer", "header", "aside", "ul"]):
        tag.decompose()
    # Alle Paragraphen sammeln die lang genug sind
    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 80]
    return "\n\n".join(paragraphs)

def chunk(text, size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        c = " ".join(words[i:i+size])
        if len(c.strip()) > 50:
            chunks.append(c)
    return chunks

for url, title, date in SPEECHES:
    print(f"Processing: {title}...")
    text = scrape(url)
    if not text:
        print(f"  ⚠️  Nothing scraped")
        continue
    chunks = chunk(text)
    print(f"  {len(chunks)} chunks")
    for c in chunks:
        vec = model.encode(c).tolist()
        supabase.table("obama_speeches").insert({
            "title": title, "date": date,
            "chunk": c, "embedding": vec
        }).execute()
    print(f"  ✓ Done")

print("\nAll speeches ingested!")

