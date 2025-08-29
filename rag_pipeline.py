import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import cohere

load_dotenv()


class RAGPipeline:
    def __init__(self):
        # pinecone setup - 
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = 'predusk-assessment' 
        
        # check if index exists, create if not
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")  
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # cohere embed-english-light-v3.0 
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = self.pc.Index(self.index_name)

        # cohere for embeddings 
        self.cohere = cohere.Client(os.getenv('COHERE_API_KEY'))
        
        # groq for llm - way faster than openai
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150):
        chunks = []
        start_pos = 0
        text_len = len(text)
        
        while start_pos < text_len:
            end_pos = min(start_pos + chunk_size, text_len)
            
            
            if end_pos < text_len:
                # Look for sentence endings within last 100 chars
                search_start = max(end_pos - 100, start_pos)
                found_sentence = False
                for i in range(end_pos, search_start, -1):
                    if text[i-1] in '.!?\n':
                        end_pos = i
                        found_sentence = True
                        break
                
                # If no sentence boundary found, avoid cutting words in half
                if not found_sentence and end_pos < text_len:
                    while end_pos > start_pos and text[end_pos] != ' ':
                        end_pos -= 1
            
            chunk_text = text[start_pos:end_pos].strip()
            if len(chunk_text) > 0:
                chunks.append(chunk_text)
            
            if end_pos >= text_len:
                break
            start_pos = end_pos - overlap
        
        # print(f"Created {len(chunks)} chunks")  # debug
        return chunks

    def _get_embedding(self, text: str):
        response = self.cohere.embed(
            texts=[text],
            model="embed-english-light-v3.0",
            input_type="search_document"
        )
        return response.embeddings[0]

    def upsert_text(self, text: str, document_id: str = 'user_text'):
        text_chunks = self._chunk_text(text)
        vector_list = []

        for idx, chunk in enumerate(text_chunks):
            embedding_vector = self._get_embedding(chunk)
            chunk_metadata = {
                'source_text': chunk, 
                'document_id': document_id,
                'chunk_index': idx,
                'source': 'user_upload',
                'title': document_id,
                'position': idx
            }
            vectorId = f'{document_id}-chunk-{idx}'
            vector_data = {'id': vectorId, 'values': embedding_vector, 'metadata': chunk_metadata}
            vector_list.append(vector_data)

        if len(vector_list) > 0:
            self.index.upsert(vectors=vector_list)
        return len(vector_list)

    def _rerank_results(self, query: str, passages: list):

        scores = []
        for passage in passages:
            # Basic keyword overlap scoring
            query_words = set(query.lower().split())
            passage_words = set(passage.lower().split())
            overlap = len(query_words.intersection(passage_words))
            score = overlap / max(len(query_words), 1)
            scores.append(score)
        return scores

    def _get_query_embedding(self, text: str):
        response = self.cohere.embed(
            texts=[text],
            model="embed-english-light-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]

    def query(self, query: str, document_id: str = 'user_text'):
        queryEmbedding = self._get_query_embedding(query)

        # retrieve more candidates for reranking
        search_results = self.index.query(
            vector=queryEmbedding,
            top_k=10,
            include_metadata=True,
            filter={'document_id': document_id}
        )
        
        matches = search_results['matches']
        if not matches:
            return "Sorry, I couldn't find relevant information to answer your question.", []

        # rerank the results
        passages = [match['metadata']['source_text'] for match in matches]
        rerank_scores = self._rerank_results(query, passages)
        
        # combine and sort by rerank score
        scored_results = []
        for i, score in enumerate(rerank_scores):
            scored_results.append((score, matches[i]))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # build context from top 3
        context_text = ''
        citation_list = []
        top_results = scored_results[:3]
        
        for idx, (score, match) in enumerate(top_results):
            citation_num = idx + 1
            source_text = match['metadata']['source_text']
            context_text += f'[{citation_num}] {source_text}\n\n'
            
            citation_data = {
                'citation_num': citation_num,
                'source_text': source_text
            }
            citation_list.append(citation_data)
        
        system_prompt = f'''Answer the question using only the provided context. Be conversational and use simple analogies when helpful. Always include citations [1], [2] in your response.

If the context doesn't contain the answer, say "I don't have information about that in the provided text."

Context:
{context_text}

Question: {query}

Answer:'''

        llm_response = self.groq.chat.completions.create(
            messages=[{'role': 'user', 'content': system_prompt}],
            model='llama3-8b-8192',
        )
        final_answer = llm_response.choices[0].message.content

        return final_answer, citation_list

