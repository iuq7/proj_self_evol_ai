from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch
from config import load_config

class SentenceTransformerRetriever:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        self.documents = documents
        embeddings = self.model.encode(documents, convert_to_tensor=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.cpu().detach().numpy())

    def retrieve(self, query, n_docs=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, n_docs)
        return [self.documents[i] for i in indices[0]]

class Reranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents):
        """
        Reranks the retrieved documents based on the query using a cross-encoder model.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return [doc for _, doc in sorted(zip(scores, documents), reverse=True)]

class RedactionFilter:
    def redact(self, text):
        """
        Redacts sensitive information from the text.
        This is a placeholder for a more sophisticated redaction mechanism.
        """
        # Simple redaction of email addresses
        return text.replace("email@example.com", "[REDACTED]")

class MemoryPolicyEngine:
    def __init__(self):
        self.short_term_memory = []  # Stores recent interactions
        self.long_term_memory = []   # Stores summarized or important information

    def add_to_short_term_memory(self, interaction):
        self.short_term_memory.append(interaction)
        # Simple truncation for short-term memory
        if len(self.short_term_memory) > 5:
            self.short_term_memory.pop(0)

    def summarize_and_add_to_long_term_memory(self):
        # This is a placeholder for a more sophisticated summarization and long-term memory storage
        if self.short_term_memory:
            summary = "Summary of recent interactions: " + "; ".join(self.short_term_memory)
            self.long_term_memory.append(summary)
            self.short_term_memory = []

    def retrieve_from_memory(self, query):
        # Simple retrieval from long-term memory based on keyword matching
        relevant_memories = [mem for mem in self.long_term_memory if query.lower() in mem.lower()]
        return " ".join(relevant_memories)

class RAGBaseline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["generator_model"])
        self.model = AutoModelForCausalLM.from_pretrained(self.config["generator_model"], use_safetensors=True)
        self.retriever = SentenceTransformerRetriever(self.config["retriever_model"])
        self.memory_engine = MemoryPolicyEngine()
        self.reranker = Reranker(self.config["reranker_model"])
        self.redaction_filter = RedactionFilter()

        with open(self.config["dummy_documents_path"], "r") as f:
            dummy_documents = f.read().splitlines()
        self.retriever.build_index(dummy_documents)

    def retrieve_and_generate(self, query):
        # Add query to short-term memory
        self.memory_engine.add_to_short_term_memory(query)

        # Retrieve relevant memories
        context_from_memory = self.memory_engine.retrieve_from_memory(query)

        # Combine query with memory context
        full_query = f"{context_from_memory} {query}".strip()

        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(full_query, n_docs=self.config["n_docs"])

        # Rerank documents
        reranked_docs = self.reranker.rerank(full_query, retrieved_docs)

        # Redact sensitive information
        redacted_docs = [self.redaction_filter.redact(doc) for doc in reranked_docs]

        top_doc = redacted_docs[0] if redacted_docs else ""
        final_query = f"{top_doc} {query}"

        inputs = self.tokenizer(final_query, return_tensors="pt")
        generated = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        # Add response to short-term memory
        self.memory_engine.add_to_short_term_memory(response)

        # Periodically summarize and move to long-term memory
        if len(self.memory_engine.short_term_memory) % 3 == 0:
            self.memory_engine.summarize_and_add_to_long_term_memory()

        return response

if __name__ == "__main__":
    config = load_config()["dynamic_context"]
    config["dummy_documents_path"] = "data/dummy_documents.txt"
    rag = RAGBaseline(config)
    query1 = "who created the first computer, my email is email@example.com"
    response1 = rag.retrieve_and_generate(query1)
    print(f"Query: {query1}")
    print(f"Response: {response1}")

    query2 = "what was its purpose"
    response2 = rag.retrieve_and_generate(query2)
    print(f"Query: {query2}")
    print(f"Response: {response2}")

    query3 = "tell me more about it"
    response3 = rag.retrieve_and_generate(query3)
    print(f"Query: {query3}")
    print(f"Response: {response3}")

    print("\nLong-term memory:")
    for mem in rag.memory_engine.long_term_memory:
        print(mem)
