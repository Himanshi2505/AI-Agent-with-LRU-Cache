import json
from transformers import pipeline
from lru_cache import LRUCache
from sentence_transformers import SentenceTransformer, util
import torch


class AIAgent:
    def __init__(self, data_file, cache_capacity=3):
        self.data_file = data_file
        self.cache = LRUCache(cache_capacity)
        self.data = self.load_data()
        self.nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
        self.labels = ["average transaction amount", "total transaction amount"]
        self.previous_queries = []  
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  
        self.similarity_threshold = 0.85  
        self.cached_queries = []  
        self.cached_query_embeddings = []  

    def load_data(self):
        
        try:
            with open(self.data_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"File {self.data_file} not found. Please ensure it exists in the directory.")
        except json.JSONDecodeError:
            raise Exception(f"File {self.data_file} is not a valid JSON file.")

    def _get_average_transaction_amount(self):
        
        averages = {}
        for client in self.data["clients"]:
            total_amount = 0
            transaction_count = 0
            for year in client:
                if year.isdigit(): 
                    transactions = client[year]["transactions"]
                    total_amount += sum(t["amount"] for t in transactions)
                    transaction_count += len(transactions)
            if transaction_count > 0:
                averages[client["client_name"]] = round(total_amount / transaction_count, 2)
        return averages

    def _get_total_transaction_amount(self):
        
        totals = {}
        for client in self.data["clients"]:
            total_amount = 0
            for year in client:
                if year.isdigit():  
                    transactions = client[year]["transactions"]
                    total_amount += sum(t["amount"] for t in transactions)
            totals[client["client_name"]] = round(total_amount, 2)
        return totals

    def _get_similar_query(self, query):
       
        if not self.cached_query_embeddings:  
            return None

        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        cached_embeddings = torch.stack(self.cached_query_embeddings)

       
        similarities = util.cos_sim(query_embeddings, cached_embeddings)[0]
        max_similarity, max_index = torch.max(similarities, dim=0)

        if max_similarity.item() > self.similarity_threshold:
            return self.cached_queries[max_index.item()]
        return None

    def process_query(self, query):
        similar_query = self._get_similar_query(query)
        if similar_query:
            cached_response = self.cache.get(similar_query)
            print(f"Cache hit for similar query: {similar_query}")
            return f"Cached Response: {cached_response}"

       
        self.previous_queries.append(query)
        if len(self.previous_queries) > 5:  # Limit context size
            self.previous_queries.pop(0)

       
        classification = self.nlp(query, self.labels)
        intent = classification["labels"][0]

        if intent == "average transaction amount":
            response = self._get_average_transaction_amount()
        elif intent == "total transaction amount":
            response = self._get_total_transaction_amount()
        else:
            response = "Query not understood. Please try a different query."

      
        self.cache.put(query, response)
        self.cached_queries.append(query)
        self.cached_query_embeddings.append(self.model.encode(query, convert_to_tensor=True))

        print(f"Cache updated with query: {query}")
        return f"Generated Response: {response}"

    def follow_up_query(self, follow_up):
       
        if not self.previous_queries:
            return "No context available for follow-up. Please ask a main query first."

     
        summary = f"In your previous queries, you asked about: {', '.join(self.previous_queries)}."
        return f"Follow-Up Response: {follow_up}\n{summary}"


if __name__ == "__main__":
   
    data_file_path = "Intership-data.json"  

    
    agent = AIAgent(data_file=data_file_path, cache_capacity=3)

    # Example queries
    print(agent.process_query("What is the average transaction amount?"))
    print(agent.process_query("What is the total transaction amount?"))
    print(agent.process_query("What is the average transaction amount?"))  # Cached response
  
    # Demonstrate cache eviction
    print(agent.process_query("Show total transactions by client."))
    print(agent.process_query("Display all transaction records."))  # Cache eviction happens here

    # Context-aware follow-up query
    print(agent.follow_up_query("Can you summarize the last transactions?"))
