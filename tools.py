import json
import os
import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer
from crewai.tools import tool


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")
ORDERS_PATH = os.path.join(BASE_DIR, "data", "orders.xlsx")


# ============================================================
# LOAD COMPANY KNOWLEDGE
# ============================================================

print("Loading company knowledge...")

index = faiss.read_index(FAISS_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Change this model if your embeddings were created
# using a different embedding model.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print(f"Loaded {len(chunks)} knowledge chunks.")


# ============================================================
# COMPANY KNOWLEDGE TOOL
# ============================================================

@tool("company_knowledge_search")
def company_knowledge_search(query: str) -> str:
    """
    Search the company's internal knowledge base.

    Use this tool for questions about company policies,
    products, services, returns, refunds, shipping policies,
    troubleshooting, FAQs, and other company information.

    Returns relevant knowledge chunks with their source
    information.
    """

    try:
        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        # Make sure the query has the same dimension
        # and data type expected by FAISS.
        query_embedding = query_embedding.astype("float32")

        distances, indices = index.search(query_embedding, 5)

        results = []

        for distance, idx in zip(distances[0], indices[0]):

            if idx < 0 or idx >= len(chunks):
                continue

            chunk = chunks[idx]

            if isinstance(chunk, dict):
                text = chunk.get("text", "")

                metadata = chunk.get("metadata", {})

                source = metadata.get(
                    "source_filename",
                    metadata.get("source", "Unknown")
                )

                page = metadata.get(
                    "page_number",
                    metadata.get("page", "Unknown")
                )

                results.append(
                    f"""
Source: {source}
Page: {page}

Content:
{text}
"""
                )

            else:
                results.append(str(chunk))

        if not results:
            return "No relevant company information was found."

        return "\n\n".join(results)

    except Exception as e:
        return f"Knowledge search failed: {str(e)}"


# ============================================================
# LOAD ORDER DATABASE
# ============================================================

print("Loading order database...")

orders_df = pd.read_excel(ORDERS_PATH)

# Normalize column names
orders_df.columns = [
    str(column).strip()
    for column in orders_df.columns
]

print(f"Loaded {len(orders_df)} orders.")


# ============================================================
# ORDER LOOKUP TOOL
# ============================================================

@tool("order_lookup")
def order_lookup(order_id: str) -> str:
    """
    Look up an order using its Order ID.

    Use this tool for questions about order status,
    order details, products, order dates, and shipping
    information.

    Example:
    ORD-2026-1001
    """

    try:

        order_id = order_id.strip()

        matches = orders_df[
            orders_df["Order ID"]
            .astype(str)
            .str.strip()
            .str.upper()
            == order_id.upper()
        ]

        if matches.empty:
            return (
                f"No order was found with Order ID "
                f"{order_id}."
            )

        order = matches.iloc[0]

        return f"""
Order ID: {order["Order ID"]}
Customer Name: {order["Customer Name"]}
Contact Number: {order["Contact Number"]}
Product: {order["Product"]}
Order Date: {order["Order Date"]}
Shipping Address: {order["Shipping Address"]}
Status: {order["Status"]}
"""

    except Exception as e:
        return f"Order lookup failed: {str(e)}"
