import streamlit as st
import re
import os
import numpy as np
from pypdf import PdfReader

# --- Import retrieval and embedding libraries ---
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

# --- Import QA pipeline ---
from transformers import pipeline

# =============================================================
# Component 1: Data Collection & Preprocessing
# =============================================================

def load_pdf_text(file_path: str) -> str:
    """Extract raw text from a PDF file."""
    if not os.path.exists(file_path):
        return ""
    reader = PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(pages)

def normalize_text(raw_text: str) -> str:
    """Clean and normalize text by collapsing whitespace."""
    return re.sub(r"\s+", " ", raw_text).strip()

def create_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

# =============================================================
# Component 2: Basic RAG Implementation (Stage-1 Retrieval)
# =============================================================
# Here we use BM25 to quickly retrieve a candidate set.

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_ID = "deepset/roberta-base-squad2"

class BasicBM25Retriever:
    def __init__(self, docs: list):
        self.docs = docs
        tokenized_docs = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, top_n: int = 20) -> list:
        """Retrieve top_n candidate indices using BM25."""
        scores = self.bm25.get_scores(query.split())
        indices = np.argsort(scores)[-top_n:]
        return indices, scores

# =============================================================
# Component 3: Advanced RAG Implementation (Multi-Stage Retrieval)
# =============================================================
# In stage 2, we re-rank the BM25 candidate set using dense retrieval.
class MultiStageRetriever:
    def __init__(self, docs: list, embed_model_id: str = EMBED_MODEL_ID):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model_id)
        # Precompute dense embeddings for all documents/chunks
        self.dense_embeddings = self.embedder.encode(docs, show_progress_bar=True)
        self.dense_embeddings = np.array(self.dense_embeddings, dtype="float32")
        # BM25 retriever (stage 1)
        self.bm25_retriever = BasicBM25Retriever(docs)
    
    def multi_stage_search(self, query: str, stage1_top: int = 20, final_top: int = 5) -> list:
        """
        Multi-stage retrieval:
        Stage 1: Use BM25 to retrieve top stage1_top candidate indices.
        Stage 2: Re-rank the candidate set using dense similarity.
        Returns a list of (doc, score) tuples for the final top candidates.
        """
        candidate_indices, bm25_scores = self.bm25_retriever.retrieve(query, top_n=stage1_top)
        # Gather candidate embeddings and texts
        candidate_embeddings = self.dense_embeddings[candidate_indices]
        candidate_texts = [self.docs[i] for i in candidate_indices]
        
        # Build a temporary FAISS index for the candidate set
        dim = candidate_embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(candidate_embeddings)
        
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")
        dists, temp_indices = temp_index.search(query_embedding, final_top)
        results = []
        for rank, local_idx in enumerate(temp_indices[0]):
            global_idx = candidate_indices[local_idx]
            # Convert distance to a positive similarity score
            score = 1.0 / (1.0 + dists[0, rank])
            results.append((self.docs[global_idx], score))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results


# =============================================================
# Component 4: UI Development (Streamlit)
# =============================================================
# The UI accepts user queries and displays the answer along with debug info.

# =============================================================
# Component 5: Guard Rail Implementation
# =============================================================
# Input-side: Block queries containing restricted keywords.
RESTRICTED_TERMS = ["ssn", "social security number", "supplier discount", "secret code", "internal phone"]

def is_query_allowed(query: str) -> bool:
    """Return False if query contains any restricted term."""
    lower = query.lower()
    for term in RESTRICTED_TERMS:
        if term in lower:
            return False
    return True

# Output-side: Sanitize any answer that reveals sensitive details.
def sanitize_answer(answer: str) -> str:
    """Redact confidential discount terms from the output."""
    patterns = [
        r"15%\s*volume discount for orders\s*>?\s*1,000 units",
        r"12%\s*rebate on monthly orders",
        r"10%\s*discount upon 6-month contract lock"
    ]
    for pattern in patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return "Restricted information cannot be provided."
    return answer

# =============================================================
# Component 6: QA Model Integration & Testing
# =============================================================
# We integrate a QA model to extract the final answer from the retrieved chunks.
class QAModule:
    def __init__(self, qa_model_id: str = QA_MODEL_ID):
        self.qa_pipeline = pipeline("question-answering", model=qa_model_id, tokenizer=qa_model_id)
    
    def answer_question(self, question: str, candidate_chunks: list) -> dict:
        best_result = {"answer": "", "score": 0.0}
        for chunk, ret_score in candidate_chunks:
            try:
                result = self.qa_pipeline({"question": question, "context": chunk})
                # Combine QA score with a fraction of retrieval score (adjust weight as needed)
                combined = result["score"] + (0.2 * ret_score)
                if combined > best_result["score"]:
                    best_result = result
                    best_result["score"] = combined
            except Exception as e:
                st.write("QA error:", e)
        return best_result

# =============================================================
# Main Application: Bringing All Components Together
# =============================================================
def main():
    st.title("Financial RAG System (Multi-Stage Retrieval + QA)")
    
    # --- Data Collection & Preprocessing ---
    pdf_2022_path = "financial_report_2022.pdf"
    pdf_2023_path = "financial_report_2023.pdf"
    if not (os.path.exists(pdf_2022_path) and os.path.exists(pdf_2023_path)):
        st.error("Please ensure 'financial_report_2022.pdf' and 'financial_report_2023.pdf' are in the current directory.")
        return

    raw_text_2022 = normalize_text(load_pdf_text(pdf_2022_path))
    raw_text_2023 = normalize_text(load_pdf_text(pdf_2023_path))
    # Create year-tagged chunks
    chunks_2022 = [f"Year: 2022\n{chunk}" for chunk in create_chunks(raw_text_2022)]
    chunks_2023 = [f"Year: 2023\n{chunk}" for chunk in create_chunks(raw_text_2023)]
    all_chunks = chunks_2022 + chunks_2023
    st.info("Data collection and chunking completed for financials of 2022 and 2023.")
    
    # --- Advanced RAG Implementation: Multi-Stage Retrieval ---
    multi_stage_retriever = MultiStageRetriever(all_chunks, embed_model_id=EMBED_MODEL_ID)
    
    # --- QA Model Setup ---
    qa_module = QAModule(qa_model_id=QA_MODEL_ID)
    
    st.success("System is ready. Enter your query below:")
    
    user_query = st.text_input("Enter your financial question:")
    if user_query:
        # Input-Side Guardrail
        if not is_query_allowed(user_query):
            st.warning("Your query contains restricted keywords. Access denied.")
            return
        
        # Multi-Stage Retrieval: Stage 1 (BM25) then Stage 2 (Dense re-ranking)
        retrieved_candidates = multi_stage_retriever.multi_stage_search(user_query, stage1_top=20, final_top=5)
        
        # --- Added Fix: Year Filtering ---
        # If the query mentions a specific year, filter candidate passages accordingly.
        if "2023" in user_query:
            retrieved_candidates = [cand for cand in retrieved_candidates if "Year: 2023" in cand[0]]
        elif "2022" in user_query:
            retrieved_candidates = [cand for cand in retrieved_candidates if "Year: 2022" in cand[0]]
        
        with st.expander("Debug: Retrieved Passages"):
            for passage, score in retrieved_candidates:
                st.write(f"Score: {score:.2f}\nPassage: {passage}\n{'-'*50}")
        
        # QA: Extract answer from retrieved passages
        qa_result = qa_module.answer_question(user_query, retrieved_candidates)
        final_answer = qa_result["answer"].strip()
        if not final_answer or qa_result["score"] < 0.1:
            final_answer = "I'm not sure."
        
        # Output-Side Guardrail: Sanitize answer if sensitive content detected
        safe_answer = sanitize_answer(final_answer)
        
        st.subheader("Answer")
        st.write(safe_answer)
        st.write(f"**Confidence:** {qa_result['score']:.2f}")

if __name__ == "__main__":
    main()
