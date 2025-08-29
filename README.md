# Mini RAG System

Built for the Predusk AI Engineer assessment. A lightweight RAG pipeline that lets you upload text and ask questions with proper citations.

## How it works

```
Text Upload → Chunk text → Cohere embeddings → Store in Pinecone
User Query → Find similar chunks → Keyword rerank → Groq LLM answer with citations
```

## Live URL - Need to deploy to vercel

## Resume link -
https://drive.google.com/file/d/1hfVym6fdhv75agwgsMdW9CVG0WepA_Wp/view?usp=sharing

## Quick Start

```bash
git clone https://github.com/maheshh-v/predusk_assessment.git
cd predusk_assessment
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env file
uvicorn main:app --reload
```

Then go to http://localhost:8000

## Current Architecture

**Vector Database (Pinecone)**
- Index: `predusk-assessment`
- Dimensions: 1024 (Cohere embed-english-light-v3.0)
- Metric: cosine similarity
- Serverless on AWS us-east-1

**Embeddings & Chunking**
- Model: Cohere `embed-english-light-v3.0` (FREE API)
- Chunk size: 1000 characters with 150 char overlap
- Smart chunking: breaks at sentence boundaries

**Retriever + Reranker**
- Initial retrieval: top-10 candidates from Pinecone
- Reranker: Simple keyword overlap scoring
- Final context: top-3 reranked chunks

**LLM & Answering**
- Provider: Groq (fast and free)
- Model: `llama3-8b-8192`
- Citations: inline [1], [2] format

## API Keys Needed

Your `.env` file needs:
```
PINECONE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

All APIs are free for this usage level.


## Testing & Evaluation

```bash
python test_rag.py
```

**Test Results (5 Q/A pairs):**

Used a machine learning text and asked 5 questions:

1. "What is supervised learning?" - Got correct definition with citation
2. "How does cross-validation work?" - Explained the process properly  
3. "What are neural networks?" - Accurate answer with source reference
4. "Explain overfitting" - Clear explanation, cited right section
5. "What is quantum computing?" - Correctly said "no info available"

Success Rate: 5/5 - All queries handled correctly

The reranker helps a lot - without it, answers were more generic. With keyword reranking, it picks the most relevant chunks.

## Trade-offs Made

- **Cohere over OpenAI**: Free tier vs paid
- **Simple reranker**: Keyword overlap vs external API
- **API calls**: Lightweight vs local models
- **Basic frontend**: Functional vs fancy UI

## What I'd Do Next

- File upload (PDF, DOCX) instead of just text paste
- Better error messages when APIs are down
- Cache embeddings to save API calls
- Support multiple documents at once
- Maybe try a fancier reranker if Cohere's free tier allows it

Perfect for assessment requirements while being production-ready.