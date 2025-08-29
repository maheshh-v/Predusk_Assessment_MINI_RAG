# Mini RAG System

Built this for the Predusk AI Engineer assessment. It's a straightforward RAG pipeline that lets you upload text and ask questions about it with proper citations.

**Live Demo**: 

**Resume**: https://drive.google.com/file/d/1hfVym6fdhv75agwgsMdW9CVG0WepA_Wp/view?usp=sharing

## How it works

```
Text Upload → Split into chunks → Create embeddings → Store in Pinecone
User Query → Find similar chunks → Rerank results → Generate answer with citations
```

Pretty standard RAG flow, nothing fancy but it gets the job done.

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

## Architecture Details

**Vector Database (Pinecone)**
- Index: `predusk-assessment`
- Dimensions: 384 (matches the embedding model)
- Metric: cosine similarity
- Serverless on AWS us-east-1
- Upsert strategy: Replace existing chunks when re-uploading same doc

**Embeddings & Chunking**
- Model: `all-MiniLM-L6-v2` from SentenceTransformers
- Chunk size: 1000 characters with 150 char overlap
- Smart chunking: tries to break at sentence boundaries, avoids cutting words
- Metadata stored: source text, doc ID, chunk position for citations

**Retriever + Reranker**
- Initial retrieval: top-10 candidates from Pinecone
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Final context: top-3 reranked chunks

**LLM & Answering**
- Provider: Groq (fast and free)
- Model: `llama3-8b-8192`
- Citations: inline [1], [2] format with source snippets below
- Handles "no answer" cases gracefully

**Frontend**
- Simple HTML/JS - no frameworks needed
- Shows processing time and chunk counts
- Basic error handling

## Testing Results

Tested with a machine learning document and 5 questions:

1. "What is supervised learning?" - Got definition with [1] citation
2. "How does cross-validation work?"  - Explained process with citations
3. "What are neural networks?" - Accurate answer with [2] reference
4. "Explain overfitting"  - Clear explanation, cited relevant section
5. "What is quantum computing?"  - Correctly said "no info in provided text"

**Success rate: 5/5** - All queries handled properly

## Services & Costs

- **Pinecone**: Free tier (100k vectors)
- **Groq**: Free tier (pretty generous limits)
- **HuggingFace**: Free models (SentenceTransformers, CrossEncoder)



## Environment Setup

Your `.env` file needs:
```
PINECONE_API_KEY=pk-...
GROQ_API_KEY=gsk_...
```

## Deployment

Should work on any platform that supports Python. API keys stay server-side.

Tested locally but ready for Render/Railway/Fly deployment.

## Remarks

**note:**
- While the requirement suggested OpenAI/Cohere/Voyag.., I chose HuggingFace's all-MiniLM-L6-v2 to demonstrate cost optimization and production mindset 
-used FastAPI instead of Streamlit because it’s production-ready, scalable for multiple users, and aligns with real-world backend engineering practices rather than just being a demo UI/.



**Trade-offs made:**
- Kept chunking simple (characters vs tokens) - good enough for this scope
- No file upload UI - just paste text (saves time, meets requirements)
- Basic frontend styling - focused on functionality
- No caching - would add Redis for production

**If I had more time:**
- File upload support (PDF, DOCX)
- Better error messages and loading states
- Token usage tracking and cost estimates
- Chunk visualization for debugging
- Support for multiple documents

