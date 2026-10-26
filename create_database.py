import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("deeplearning.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma(
    collection_name="deeplearning",
    embedding_function=embedding_model,
    persist_directory="chroma_db",
)

# Resume: chunks are added in order, so the count tells us where we stopped
already_done = vectorstore._collection.count()
print(f"Already in database: {already_done} chunks")

if already_done >= len(chunks):
    print("Nothing left to embed.")
    raise SystemExit

BATCH_SIZE = 90

for i in range(already_done, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]

    for attempt in range(4):
        try:
            vectorstore.add_documents(batch)
            break
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                print("Daily quota exhausted. Run this script again after the quota resets.")
                raise SystemExit
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                wait = 30 * (attempt + 1)
                print(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Batch starting at chunk {i} failed after retries")

    print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    if i + BATCH_SIZE < len(chunks):
        time.sleep(61)

print("Database created and saved successfully!")