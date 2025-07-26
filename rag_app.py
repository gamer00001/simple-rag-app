import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"&|,|/|\\", " and ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load data
df = pd.read_csv("Outage Handling - handling_data.csv")
df["Outage_norm"] = df["Outage"].apply(normalize_text)
Outage_texts = df["Outage_norm"].tolist()

# Embed outage descriptions
Outage_embeddings = embedding_model.encode(Outage_texts)

# Create FAISS index
index = faiss.IndexFlatL2(Outage_embeddings[0].shape[0])
index.add(np.array(Outage_embeddings))

# Query Gemini with matched entries
# def query_gemini_multi(results, user_query):
#     entries = "\n".join([
#         f"- Outage: {r['Outage']}\n  Impact: {r['impact']}\n  Handling: {r['handling']}"
#         for r in results
#     ])

#     prompt = f"""
# You are a system operations assistant. The user asked:

# "{user_query}"

# Here are the matched outages and their details:

# {entries}

# format the entries in order form & Don't add any note or recomendation or suggestion.
# """.strip()

#     messages = [HumanMessage(content=prompt)]
#     response = llm.invoke(messages)
#     return response.content

def query_gemini_multi(results, user_query):
    # Combine outages
    outages = [r['Outage'] for r in results]
    impacts = [r['impact'] for r in results]
    handlings = [r['handling'] for r in results]

    # Join outages in order
    combined_outage = " and ".join(outages)

    # Merge impacts line by line
    combined_impact = "\n".join(f"- {impact.strip()}" for impact in impacts if impact.strip())

    # Split handlings into individual actions
    all_handlings = []
    for h in handlings:
        parts = re.split(r"\n|;| and ", h)
        all_handlings.extend([p.strip() for p in parts if p.strip()])

    # Remove duplicates and preserve order
    unique_handlings = list(dict.fromkeys(all_handlings))
    formatted_handling = "\n".join(f"- {h}" for h in unique_handlings)

    # Final output
    return f"""**Outage:** {combined_outage}

**Impact:**
{combined_impact}

**Handling:**
{formatted_handling}
"""

# Semantic search + LLM response
def search_and_respond(user_query, top_k=1):
    user_query_norm = normalize_text(user_query)
    user_embedding = embedding_model.encode([user_query_norm])
    distances, indices = index.search(np.array(user_embedding), k=top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "Outage": row["Outage"],
            "impact": row["Impact"],
            "handling": row["Handling"]
        })

    response = query_gemini_multi(results, user_query)
    return results, response

# Streamlit UI
st.title("Outage Handler")

query = st.text_input("Describe the issue (e.g. 'uet3 and uet4 down'):")

if query:
    matched_results, gemini_response = search_and_respond(query)

    st.subheader("Matched Outages")
    # for r in matched_results:
    # st.markdown(f"**Outage**: {matched_results[0]['Outage']}")
    # st.markdown(f"**Impact**: {matched_results[0]['impact']}")
    # st.markdown(f"**Handling**: {matched_results[0]['handling']}")
    # st.markdown("---")

    # st.subheader("Gemini Summary")
    # st.write(gemini_response)
    st.markdown(gemini_response)

