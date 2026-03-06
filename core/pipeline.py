"""
Pipeline Orchestrator - Groq version (free, fast, Llama 3)
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from agents.planner import QueryPlannerAgent
from agents.retriever import RetrieverAgent
from agents.critic import CriticAgent
from agents.synthesizer import SynthesizerAgent

load_dotenv()


class GroqModel:
    """Wrapper so all agents work with Groq seamlessly."""
    def __init__(self, client, model_name="llama-3.1-8b-instant"):
        self.client = client
        self.model_name = model_name

    def generate_content(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        text = response.choices[0].message.content
        return type('R', (), {'text': text})()


class FinSightPipeline:
    def __init__(self, index_dir: str = "data/index"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        client = Groq(api_key=api_key)
        self.llm = GroqModel(client)

        self.processor = DocumentProcessor(chunk_size=500, overlap=100)
        self.vector_store = VectorStore(index_dir=index_dir)

        self.planner = QueryPlannerAgent(self.llm)
        self.retriever = RetrieverAgent(self.vector_store, top_k=15)
        self.critic = CriticAgent(self.llm)
        self.synthesizer = SynthesizerAgent(self.llm)

        self.indexed_docs = []
        if self.vector_store.load():
            self.indexed_docs = self.vector_store.stats()["documents"]

    def index_document(self, pdf_path: str) -> dict:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        print(f"\nIndexing: {path.name}")
        chunks = self.processor.process(pdf_path)
        self.vector_store.add_chunks(chunks)
        self.vector_store.save()
        if path.name not in self.indexed_docs:
            self.indexed_docs.append(path.name)
        return {
            "filename": path.name,
            "chunks": len(chunks),
            "total_indexed": len(self.vector_store.chunks)
        }

    def query(self, question: str) -> dict:
        if self.vector_store.is_empty():
            return {"error": "No documents indexed. Please upload a PDF first.", "answer": None}

        start = time.time()
        print(f"\n[1/4] QueryPlanner running...")
        sub_questions = self.planner.run(question)
        print(f"  Sub-questions: {sub_questions}")

        print(f"[2/4] Retriever running...")
        results, context = self.retriever.run(sub_questions)
        print(f"  Retrieved {len(results)} unique chunks")

        print(f"[3/4] Critic running...")
        verdict = self.critic.run(question, context, results)
        print(f"  Sufficient: {verdict.sufficient} | Confidence: {verdict.confidence}")

        print(f"[4/4] Synthesizer running...")
        result = self.synthesizer.run(question, context, results, verdict, sub_questions)

        elapsed = round(time.time() - start, 2)
        print(f"\nDone in {elapsed}s")

        return {
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "sub_questions": result.sub_questions,
            "verdict": result.verdict,
            "elapsed_seconds": elapsed
        }

    def get_stats(self) -> dict:
        return self.vector_store.stats()
