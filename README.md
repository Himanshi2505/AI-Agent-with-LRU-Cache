# **AI Agent with LRU-Based Caching**

This project implements an AI-powered agent that efficiently responds to user queries by leveraging an LRU (Least Recently Used) caching mechanism. The agent is designed to process transaction-related queries, retrieve or compute the relevant data, and store query results in a cache to ensure faster responses for repeated queries.

---

## **Features**

### **1. AI Query Processing**
- The agent processes natural language queries such as:
  - "What is the average transaction amount?"
  - "What is the total transaction amount?"
- It dynamically classifies queries and generates appropriate responses based on the provided dataset (`Intership-data.json`).

### **2. LRU Caching Implementation**
- Uses an LRU cache to store query results:
  - If a query is repeated, the result is returned from the cache without recomputation.
  - When the cache is full, the least recently used query-result pair is evicted.
- The cache ensures performance optimization with `O(1)` operations for insertion, lookup, and eviction.

### **3. Performance Optimization**
- The caching system is implemented using:
  - A hash map (dictionary) for fast lookups.
  - A doubly linked list to maintain query usage order.

### **4. Advanced Prompting Techniques**
- **Context-Aware Prompts**: Tracks previous queries to provide more relevant responses for follow-up queries.
- **Few-Shot Learning**: Improves query understanding and response accuracy using dynamic query classification (`facebook/bart-large-mnli` model).
- **Semantic Query Matching**: Uses semantic similarity (via `sentence-transformers`) to find and reuse responses for queries similar to those already processed.

### **5. Robust Handling**
- Validates and processes input queries dynamically.
- Handles missing or invalid data gracefully with informative error messages.

---

## **Setup Instructions**

1. Clone the repositpory.
2. Install dependencies.
3. To run LRU Cache inplementation run the command python lru_cache.py
4. To run Ai agent run the command python ai_agent.py
