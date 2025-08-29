#!/usr/bin/env python3

import time
from rag_pipeline import RAGPipeline

def test_rag_pipeline():
    print("Testing RAG Pipeline...")
    
    # Initialize pipeline
    print("1. Initializing RAG pipeline...")
    try:
        rag = RAGPipeline()
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False
    
    # Test text for upload
    test_text = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. 
    
    Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The goal is to map input features to correct output labels.
    
    Cross-validation is a technique used to assess how well a machine learning model will generalize to new data. It involves splitting the dataset into multiple folds and training/testing on different combinations.
    
    Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.
    
    Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor performance on new data.
    """
    
    # Test upload
    print("2. Testing text upload...")
    try:
        start_time = time.time()
        chunk_count = rag.upsert_text(test_text, "test_doc")
        upload_time = time.time() - start_time
        print(f"Uploaded {chunk_count} chunks in {upload_time:.2f}s")
    except Exception as e:
        print(f"Upload failed: {e}")
        return False
    
    # Test queries
    test_queries = [
        "What is supervised learning?",
        "How does cross-validation work?", 
        "What are neural networks?",
        "Explain overfitting",
        "What is quantum computing?"  # Should return no answer
    ]
    
    print("3. Testing queries...")
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        try:
            start_time = time.time()
            answer, citations = rag.query(query, "test_doc")
            query_time = time.time() - start_time
            
            print(f"   Response time: {query_time:.2f}s")
            print(f"   Answer: {answer[:100]}...")
            print(f"   Citations: {len(citations)} sources")
            
            if answer and "sorry" not in answer.lower():
                success_count += 1
                print("   Query successful")
            else:
                print("   No relevant answer found")
                
        except Exception as e:
            print(f"   Query failed: {e}")
    
    print(f"\nTest Results: {success_count}/{len(test_queries)} queries successful")
    
    if success_count >= 4:  # Allow 1 failure for the quantum computing question
        print("RAG Pipeline is working correctly!")
        return True
    else:
        print("RAG Pipeline needs debugging")
        return False

if __name__ == "__main__":
    test_rag_pipeline()