import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv
import re

import faiss
import pickle

index = faiss.read_index("outage_index.faiss")

with open("outage_metadata.pkl", "rb") as f:
    df = pickle.load(f)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[&/\\,;-]", " ", text)  # replace common delimiters with space
    text = re.sub(r"([a-z]+)(\d+)", r"\1 \2", text)  # add space between unit and number
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text


def extract_units(text):
    text = normalize_text(text)

    # Match combined formats like "uet12" or "shr12"
    combined_matches = re.findall(r"(uet|shr)\s*\d{1,2}", text)
    combined_units = re.findall(r"(uet|shr)\s*\d{1,2}", text)
    explicit_units = re.findall(r"(uet|shr)\s*\d{1,2}", text)
    
    # Extract full matches like "uet12", then split into "uet 12"
    explicit_units = []
    for match in re.findall(r"(uet|shr)\s*\d{1,2}", text):
        unit = re.search(rf"({match})\s*(\d{{1,2}})", text)
        if unit:
            explicit_units.append(f"{match} {unit.group(2)}")

    # Match grouped formats like "uet 1 2" or "shr 3 4"
    group_units = []
    for prefix in ["uet", "shr"]:
        match = re.search(rf"{prefix}\s*((?:\d{{1,2}}\s*)+)", text)
        if match:
            nums = re.findall(r"\d{1,2}", match.group(1))
            group_units.extend([f"{prefix} {n}" for n in nums])

    all_units = set(explicit_units + group_units)
    return sorted(all_units)



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
    user_units = extract_units(user_query_norm)

    user_embedding = embedding_model.encode([user_query_norm])
    distances, indices = index.search(np.array(user_embedding), k=top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        outage_units = extract_units(row["Outage"])
        overlap = len(set(user_units) & set(outage_units))

        results.append({
            "Outage": row["Outage"],
            "impact": row["Impact"],
            "handling": row["Handling"],
            "score": overlap  # unit overlap score
        })

    # Fallback: Prefer rows with highest overlap
    results.sort(key=lambda r: r["score"], reverse=True)
    top_results = results[:1] if results and results[0]["score"] > 0 else results[:1]

    response = query_gemini_multi(top_results, user_query)
    return top_results, response


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

