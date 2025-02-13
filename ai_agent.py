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
        self.previous_queries = []  # To maintain context for follow-up queries
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # For semantic similarity
        self.similarity_threshold = 0.85  # Threshold for matching similar queries
        self.cached_queries = []  # Track cached queries
        self.cached_query_embeddings = []  # Track embeddings of cached queries

    def load_data(self):
        """Load the dataset from the JSON file."""
        try:
            with open(self.data_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"File {self.data_file} not found. Please ensure it exists in the directory.")
        except json.JSONDecodeError:
            raise Exception(f"File {self.data_file} is not a valid JSON file.")

    def _get_average_transaction_amount(self):
        """Calculate the average transaction amount for each client."""
        averages = {}
        for client in self.data["clients"]:
            total_amount = 0
            transaction_count = 0
            for year in client:
                if year.isdigit():  # Check if the key is a year
                    transactions = client[year]["transactions"]
                    total_amount += sum(t["amount"] for t in transactions)
                    transaction_count += len(transactions)
            if transaction_count > 0:
                averages[client["client_name"]] = round(total_amount / transaction_count, 2)
        return averages

    def _get_total_transaction_amount(self):
        """Calculate the total transaction amount for each client."""
        totals = {}
        for client in self.data["clients"]:
            total_amount = 0
            for year in client:
                if year.isdigit():  # Check if the key is a year
                    transactions = client[year]["transactions"]
                    total_amount += sum(t["amount"] for t in transactions)
            totals[client["client_name"]] = round(total_amount, 2)
        return totals

    def _get_similar_query(self, query):
        """Find the most similar query from cached queries using cosine similarity."""
        if not self.cached_query_embeddings:  # Check if there are no cached embeddings
            return None

        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        cached_embeddings = torch.stack(self.cached_query_embeddings)

        # Compute cosine similarity
        similarities = util.cos_sim(query_embeddings, cached_embeddings)[0]
        max_similarity, max_index = torch.max(similarities, dim=0)

        # Return the most similar query if similarity is above a certain threshold
        if max_similarity.item() > self.similarity_threshold:
            return self.cached_queries[max_index.item()]
        return None

    def process_query(self, query):
        """Process a user query and return a response."""
        # Check for similar queries in the cache
        similar_query = self._get_similar_query(query)
        if similar_query:
            cached_response = self.cache.get(similar_query)
            print(f"Cache hit for similar query: {similar_query}")
            return f"Cached Response: {cached_response}"

        # Add the query to the context
        self.previous_queries.append(query)
        if len(self.previous_queries) > 5:  # Limit context size
            self.previous_queries.pop(0)

        # Classify the query dynamically
        classification = self.nlp(query, self.labels)
        intent = classification["labels"][0]

        if intent == "average transaction amount":
            response = self._get_average_transaction_amount()
        elif intent == "total transaction amount":
            response = self._get_total_transaction_amount()
        else:
            response = "Query not understood. Please try a different query."

        # Store the result in the cache
        self.cache.put(query, response)
        self.cached_queries.append(query)
        self.cached_query_embeddings.append(self.model.encode(query, convert_to_tensor=True))

        print(f"Cache updated with query: {query}")
        return f"Generated Response: {response}"

    def follow_up_query(self, follow_up):
        """Handle follow-up queries using context from previous queries."""
        if not self.previous_queries:
            return "No context available for follow-up. Please ask a main query first."

        # Summarize the context for meaningful follow-up responses
        summary = f"In your previous queries, you asked about: {', '.join(self.previous_queries)}."
        return f"Follow-Up Response: {follow_up}\n{summary}"


if __name__ == "__main__":
    # Path to the provided dataset
    data_file_path = "Intership-data.json"  # Ensure this file is in the same directory

    # Initialize the AI Agent with a cache capacity of 3
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
