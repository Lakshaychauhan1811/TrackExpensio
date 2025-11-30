import asyncio
import base64
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict

import faiss
from groq import Groq
from langgraph.graph import END, StateGraph
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def _require_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"{var} is required but missing from environment")
    return value


def extract_amount_from_text(text: str) -> Optional[float]:
    """Extract the total amount from text, prioritizing 'Total', 'Amount Due', 'Grand Total' keywords."""
    # First, try to find amounts near "Total", "Amount Due", "Grand Total", etc.
    total_patterns = [
        r"(?:total|amount\s+due|grand\s+total|final\s+amount|payable|balance|subtotal|net\s+amount)[\s:]*[₹$€£¥]?\s*([\d,]+(?:\.\d{1,2})?)",
        r"[₹$€£¥]\s*([\d,]+(?:\.\d{1,2})?)\s*(?:total|amount|due|payable)",
        r"(?:total|amount|due)[\s:]*[\d,]+(?:\.\d{1,2})?\s*[₹$€£¥]?\s*([\d,]+(?:\.\d{1,2})?)",  # Second amount after total keyword
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)
                if amount > 0:
                    return amount
            except (ValueError, IndexError):
                continue
    
    # If no total found, find all amounts and return the largest (usually the total)
    amounts = []
    # Find all amounts with currency symbols
    currency_amounts = re.findall(r'[₹$€£¥]\s*([\d,]+(?:\.\d{1,2})?)', text, re.IGNORECASE)
    for amt_str in currency_amounts:
        try:
            amount = float(amt_str.replace(",", ""))
            if amount > 0:
                amounts.append(amount)
        except ValueError:
            continue
    
    # Also find standalone large numbers (likely amounts)
    standalone_amounts = re.findall(r'\b([\d,]+(?:\.\d{1,2})?)\b', text)
    for amt_str in standalone_amounts:
        try:
            amount = float(amt_str.replace(",", ""))
            # Filter out small numbers (likely quantities, not amounts) and very large numbers (likely invoice numbers)
            if 10 <= amount <= 10000000:  # Reasonable expense range
                amounts.append(amount)
        except ValueError:
            continue
    
    if amounts:
        # Return the largest amount (usually the total)
        return max(amounts)
    
    return None


class RAGState(TypedDict):
    """State for LangGraph RAG workflow"""
    question: str
    documents: list
    context: str
    answer: str


class _SimpleRetriever:
    """Minimal FAISS-backed retriever without LangChain."""

    def __init__(self, texts: list[str], embed_model: SentenceTransformer):
        if not texts:
            raise RuntimeError("No text chunks available for retrieval")
        self.texts = texts
        self.embed_model = embed_model

        embeddings = embed_model.encode(
            texts,
            batch_size=int(os.getenv("RAG_EMBED_BATCH_SIZE", "8")),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def invoke(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        vector = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        top_k = min(k, len(self.texts))
        scores, indices = self.index.search(vector, top_k)
        docs: list[dict[str, Any]] = []
        for idx in indices[0]:
            idx_int = int(idx)
            if 0 <= idx_int < len(self.texts):
                docs.append({"text": self.texts[idx_int], "metadata": {"chunk": idx_int}})
        return docs


def _read_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        content = page.extract_text() or ""
        content = content.strip()
        if content:
            pages.append(content)
    combined = "\n\n".join(pages).strip()
    if not combined:
        raise RuntimeError("Unable to read document")
    return combined


def _split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


async def _build_retriever_from_chunks(chunks: list[str]) -> _SimpleRetriever:
    def _prepare() -> _SimpleRetriever:
        model_name = os.getenv(
            "RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embed_model = SentenceTransformer(model_name)
        return _SimpleRetriever(chunks, embed_model)

    return await asyncio.to_thread(_prepare)


async def run_receipt_rag_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Uses LangGraph + FAISS + Groq LLM (no LangChain) to extract structured fields
    from a receipt/invoice PDF or image.
    
    Enhanced with better error handling, validation, and extraction accuracy.
    """

    _require_env("GROQ_API_KEY")

    # Read and validate document
    try:
        raw_text = _read_pdf_text(file_path)
        if not raw_text or len(raw_text.strip()) < 10:
            raise ValueError("Document appears to be empty or unreadable. Please ensure the document contains text.")
    except Exception as e:
        raise ValueError(f"Failed to read document: {str(e)}. Please ensure it's a valid PDF or image file.")

    # Split into chunks with validation
    chunks = _split_text(raw_text)
    if not chunks or len(chunks) == 0:
        raise ValueError("Could not extract text chunks from document. The document may be corrupted or unreadable.")
    
    # Build retriever with error handling
    try:
        retriever = await _build_retriever_from_chunks(chunks)
    except Exception as e:
        raise ValueError(f"Failed to build document retriever: {str(e)}")
    
    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Define LangGraph nodes
    def retrieve_documents(state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        question = state["question"]
        docs = retriever.invoke(question)
        state["documents"] = docs
        state["context"] = "\n\n".join(doc["text"] for doc in docs)
        return state

    def generate_answer(state: RAGState) -> RAGState:
        """Generate answer using Groq LLM directly"""
        context = state["context"]
        question = state["question"]
        
        # Build enhanced prompt with better instructions
        prompt = f"""You are an expert finance assistant specialized in extracting structured information from receipts, invoices, and bills.

Document Context:
{context}

Task: {question}

Extract the following information with high accuracy:

1. merchant: The name of the store, restaurant, business, or service provider. Look for business names, logos, or headers.
2. amount: The TOTAL amount paid (final amount including taxes). Extract as a NUMBER, not a string. 
   CRITICAL: Look for keywords like "Total", "Amount Due", "Grand Total", "Final Amount", "Payable", "Balance", "Net Amount".
   If multiple amounts are present, ALWAYS choose the LARGEST amount that appears near these keywords.
   DO NOT extract small amounts like item prices, quantities, or partial totals. Only extract the FINAL TOTAL.
   Example: If you see "Subtotal: ₹500, Tax: ₹50, Total: ₹550", extract 550, not 500 or 50.
3. currency: Currency code (INR, USD, EUR, GBP, JPY, etc.). Look for currency symbols (₹, $, €, £, ¥) or currency codes. Default to INR if not specified.
4. category: Expense category based on merchant type and items purchased. Categories: Food, Travel, Bills, Shopping, Entertainment, Utilities, Rent, Health, Education, Other. Infer from:
   - Restaurant/cafe/food → Food
   - Hotel/flight/taxi → Travel
   - Store/retail → Shopping
   - Utility/phone/internet → Utilities
   - Rent/housing → Rent
   - Medical/pharmacy → Health
   - School/tuition → Education
   - Subscription/streaming → Entertainment
5. date: Transaction date in YYYY-MM-DD format. Look for date fields, invoice dates, or transaction dates. If not found, use today's date: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
6. notes: Additional relevant information like items purchased, invoice number, or transaction details.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no markdown, no code blocks
- All amounts must be numbers (not strings)
- Dates must be in YYYY-MM-DD format
- If information is missing, use null (not empty string)
- Be precise and accurate

Example JSON format:
{{"merchant": "Restaurant Name", "amount": 500.00, "currency": "INR", "category": "Food", "date": "2025-11-30", "notes": "Dinner with family"}}"""
        
        # Call Groq API with enhanced settings
        try:
            response = groq_client.chat.completions.create(
                model=os.getenv("GROQ_RAG_MODEL", "llama-3.3-70b-versatile"),
                messages=[
                    {"role": "system", "content": "You are an expert finance assistant. Always return valid JSON only. Never include markdown code blocks or explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent, accurate extraction
                max_tokens=500,  # Limit response length
                response_format={"type": "json_object"},  # Force JSON response if supported
            )
        except Exception as e:
            raise ValueError(f"Failed to call AI model: {str(e)}. Please check your GROQ_API_KEY and try again.")
        
        answer = response.choices[0].message.content or ""
        state["answer"] = answer
        return state

    # Build LangGraph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()

    # Run the graph
    question = (
        "Extract all information from this receipt/invoice. Return JSON with keys "
        "`merchant`, `amount`, `currency`, `category`, `date`, `notes`. "
        "If a value is missing, set it to null."
    )
    
    def _run_graph():
        initial_state = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": ""
        }
        result = app.invoke(initial_state)
        return result["answer"]
    
    answer = await asyncio.to_thread(_run_graph)
    parsed = _coerce_json_response(answer)
    
    # Validate and correct amount if it seems too small (likely wrong extraction)
    if parsed.get("amount"):
        try:
            extracted_amount = float(parsed["amount"])
            # If amount is suspiciously small (< 10), try to re-extract from raw text
            if extracted_amount < 10:
                # Re-extract using improved logic
                better_amount = extract_amount_from_text(raw_text)
                if better_amount and better_amount >= 10:
                    parsed["amount"] = better_amount
        except (ValueError, TypeError):
            pass
    
    parsed["raw_text"] = raw_text
    parsed["context"] = "\n\n".join(chunks)
    return parsed


def _coerce_json_response(raw: str) -> Dict[str, Any]:
    import json

    try:
        # Try to parse as JSON first
        parsed = json.loads(raw)
        # Ensure all required fields exist
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        return parsed
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract from text using regex
        data: Dict[str, Any] = {}
        
        # Try to find JSON block in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Extract merchant/store name
        merchant_patterns = [
            r"(?:merchant|store|restaurant|vendor|business|company)\s*[:\-]?\s*([A-Za-z0-9\s&.,'-]+)",
            r'"merchant"\s*:\s*"([^"]+)"',
            r"'merchant'\s*:\s*'([^']+)'",
        ]
        merchant = None
        for pattern in merchant_patterns:
            match = re.search(pattern, raw, re.I)
            if match:
                merchant = match.group(1).strip()
                break
        
        # Extract amount - prioritize JSON format, then smart text extraction
        amount = None
        # First try JSON format
        amount_match = re.search(r'"amount"\s*:\s*([\d,]+(?:\.\d+)?)', raw)
        if amount_match:
            try:
                amount = float(amount_match.group(1).replace(",", ""))
            except:
                pass
        
        # If not found in JSON, use smart text extraction
        if not amount or amount <= 0:
            amount = extract_amount_from_text(raw)
        
        # Extract date
        date_match = re.search(r'(20\d{2}[-/]\d{1,2}[-/]\d{1,2})', raw)
        date = date_match.group(1) if date_match else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Normalize date format
        date = date.replace('/', '-')
        
        # Extract currency
        currency_match = re.search(r'"currency"\s*:\s*"([^"]+)"', raw, re.I)
        currency = currency_match.group(1).upper() if currency_match else "INR"
        
        # Extract category
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', raw, re.I)
        category = category_match.group(1) if category_match else "Bills"
        
        # Extract notes
        notes_match = re.search(r'"notes"\s*:\s*"([^"]+)"', raw)
        notes = notes_match.group(1) if notes_match else raw[:200]
        
        data["merchant"] = merchant or "Unknown"
        data["amount"] = amount
        data["date"] = date
        data["currency"] = currency
        data["category"] = category
        data["notes"] = notes
        return data


def decode_base64_file(payload: str) -> bytes:
    return base64.b64decode(payload.split(",")[-1])

