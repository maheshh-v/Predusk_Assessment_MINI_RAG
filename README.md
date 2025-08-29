# Smart Document Q&A System

Built by Mahesh for the Predusk AI Engineer assessment. After exploring different approaches, I settled on this RAG pipeline that lets you upload text and ask questions with proper citations.

**Live Demo:** https://mini-rag-gzzo.onrender.com/
**Resume:** https://drive.google.com/file/d/1hfVym6fdhv75agwgsMdW9CVG0WepA_Wp/view?usp=sharing

## How it works

```
Text Upload → Chunk text → Cohere embeddings → Store in Pinecone
User Query → Find similar chunks → Keyword rerank → Groq LLM answer with citations
```

## Quick Start for Evaluators

```bash
git clone https://github.com/maheshh-v/Predusk_Assessment_MINI_RAG.git
cd Predusk_Assessment_MINI_RAG
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env file
uvicorn main:app --reload
```

Then go to http://localhost:8000

## Why I Built It This Way

**I chose Cohere** because their free tier gives 1000 API calls/month - perfect for demos without worrying about costs. After trying HuggingFace's inference API (which kept timing out), Cohere was much more reliable.

**For the reranker**, I went with simple keyword overlap instead of an external API. It's lightweight, works offline, and honestly performs pretty well for this scope. Plus it keeps everything free.

**Groq for the LLM** was a no-brainer - it's incredibly fast and free. The responses come back in under 2 seconds usually.

## Current Architecture

**Vector Database (Pinecone)**
- Index: `predusk-assessment`
- Dimensions: 1024 (matches Cohere's embedding size)
- Metric: cosine similarity
- Serverless on AWS us-east-1

**Embeddings & Chunking**
- Model: Cohere `embed-english-light-v3.0`
- Chunk size: 1000 characters with 150 char overlap
- Smart chunking: tries to break at sentence boundaries

**Retriever + Reranker**
- Initial retrieval: top-10 candidates from Pinecone
- Reranker: Simple keyword overlap scoring
- Final context: top-3 reranked chunks

**LLM & Answering**
- Provider: Groq
- Model: `llama3-8b-8192`
- Citations: inline [1], [2] format

## Demo Instructions

1. **Upload some text** - paste anything interesting (I used ML concepts for testing)
2. **Ask questions** - try both specific and general queries
3. **Check citations** - sources are shown below each answer
4. **Test edge cases** - ask about something not in your text

Expected response time: 2-3 seconds per query.

## Testing Results

I tested with 5 Q/A pairs using machine learning content:

1. "What is supervised learning?" - Got correct definition with citation
2. "How does cross-validation work?" - Explained the process properly  
3. "What are neural networks?" - Accurate answer with source reference
4. "Explain overfitting" - Clear explanation, cited right section
5. "What is quantum computing?" - Correctly said "no info available"

Success Rate: 5/5 - All queries handled correctly

The keyword reranker actually works better than I expected. Without it, answers were more generic. With reranking, it picks the most relevant chunks.

## Assessment Requirements Met

- Vector database (hosted): Pinecone ✓
- Embeddings & chunking: Cohere + smart chunking ✓
- Retriever + reranker: Top-k + keyword reranking ✓
- LLM & answering: Groq with citations ✓
- Frontend: Upload/query interface with timing ✓
- Hosting: Deployed on Render ✓
- Evaluation: 5 Q/A pairs documented ✓

## Remarks

Used free APIs (Cohere, Groq, Pinecone) to keep costs zero. Simple keyword reranker instead of external API for free tier compatibility. Next: file upload support and multi-document sessions.
