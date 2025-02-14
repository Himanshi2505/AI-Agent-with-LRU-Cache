
This project implements an AI-powered agent that efficiently responds to user queries by leveraging an LRU (Least Recently Used) caching mechanism. The agent is designed to process transaction-related queries, retrieve or compute the relevant data, and store query results in a cache to ensure faster responses for repeated queries.

## LRU Cache Implementation

The `lru_cache.py` file contains the implementation of an **LRU (Least Recently Used) Cache**, designed to store and retrieve frequently requested data efficiently. The cache evicts the least recently used item when it reaches its capacity.  

### Features:
- Supports O(1) operations for insertion, lookup, and eviction.
- Built using a combination of a **hashmap** and a **doubly linked list**.
- Tested with sample use cases included in the file.  

You can explore this file for understanding the fundamentals of LRU caching.

## AI Agent with LRU-Based Caching  

The `ai_agent.py` file contains an AI-powered agent that processes natural language queries, retrieves relevant data, and leverages the **LRU cache** for efficient query result storage and retrieval.

### Features:
1. **AI Query Processing**:  
   - Accepts natural language queries like:
     - "What is the average transaction amount?"
     - "What is the total transaction amount?"
   - Classifies queries using the `facebook/bart-large-mnli` model.

2. **LRU Cache Integration**:  
   - Stores query results to improve efficiency for repeated queries.
   - Automatically evicts the least recently used entry when the cache is full.

3. **Advanced Prompting Techniques**:  
   - Context-aware prompts: Maintains a history of previous queries to enhance follow-up responses.
   - Semantic similarity matching: Matches new queries with cached ones using **cosine similarity** and the `all-MiniLM-L6-v2` model for faster responses to similar queries.

4. **Performance Optimization**:  
   - Designed to ensure cache operations run in O(1) time complexity.

---

## Example Workflow

1. A user asks:  
   **"What is the average transaction amount?"**  
   - The AI processes the query and calculates the average transaction values from the dataset.  
   - The result is cached for future use.

2. The user repeats the same query:  
   **"What is the average transaction amount?"**  
   - The AI retrieves the result directly from the cache, ensuring faster response time.

3. A new query exceeds cache capacity:  
   - The least recently used entry is evicted to make room for the new query.

4. Follow-up queries:  
   The AI uses previous queries to provide context-aware responses.


## **Setup Instructions**

1. Clone the repositpory.
2. Install dependencies.
3. To run LRU Cache inplementation run the command python lru_cache.py
4. To run Ai agent run the command python ai_agent.py
