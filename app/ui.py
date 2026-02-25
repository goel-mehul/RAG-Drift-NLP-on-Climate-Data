"""
ui.py
-----
Streamlit UI for the RAG climate policy system.
Run with: streamlit run app/ui.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from pipeline import RAGPipeline, SharedResources

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Policy RAG",
    page_icon="🌍",
    layout="wide",
)

# ── Load resources once (cached so they don't reload on every interaction) ────
@st.cache_resource
def load_pipeline():
    resources = SharedResources()
    return {
        "baseline": RAGPipeline("baseline", resources),
        "hybrid":   RAGPipeline("hybrid",   resources),
        "full":     RAGPipeline("full",      resources),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/globe-showing-americas_1f30e.png", width=60)
    st.title("Climate Policy RAG")
    st.markdown("Ask questions about NYC, NY State, and US federal climate policy documents.")
    st.divider()

    pipeline_choice = st.radio(
        "Pipeline variant",
        options=["full", "hybrid", "baseline"],
        format_func=lambda x: {
            "full":     "✅ Full (hybrid + faithfulness gate)",
            "hybrid":   "🔵 Hybrid (dense + BM25, no gate)",
            "baseline": "⚪ Baseline (dense only, no gate)",
        }[x],
        index=0,
    )

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What are NYC's greenhouse gas emission targets?",
        "What does Local Law 97 require of buildings?",
        "What is the global temperature goal in the Paris Agreement?",
        "What emission targets does the CLCPA set for New York?",
        "How does the Inflation Reduction Act support clean energy?",
        "What role do environmental justice communities play in NY climate policy?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["query_input"] = ex

    st.divider()
    st.caption("Built with Llama 3.1 · FAISS · BM25 · Lexical Faithfulness Gate")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🌍 Climate Policy Question Answering")
st.markdown("Ask anything about NYC, NY State, or US federal climate policy. "
            "The system retrieves from 13 real policy documents and scores answer faithfulness.")

query = st.text_input(
    "Your question",
    placeholder="e.g. What are NYC's building emissions limits under Local Law 97?",
    key="query_input",
)

run_btn = st.button("Ask", type="primary", use_container_width=False)

if run_btn and query.strip():
    pipelines = load_pipeline()
    pipeline = pipelines[pipeline_choice]

    with st.spinner("Retrieving and generating answer..."):
        result = pipeline.run(query.strip(), top_k=5)

    # ── Answer box ────────────────────────────────────────────────────────────
    st.divider()

    if result["abstained"]:
        st.warning("⚠️ **Abstained** — The faithfulness gate determined there is insufficient evidence in the documents to answer this question confidently.")
        with st.expander("Show original (unfaithful) answer"):
            st.markdown(result.get("final_answer", ""))
    else:
        st.success("✅ Answer")
        st.markdown(result["final_answer"])

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        faith = result["faithfulness"]
        if faith is not None:
            color = "green" if faith >= 0.7 else "orange" if faith >= 0.35 else "red"
            st.metric("Faithfulness", f"{faith:.2f}")
        else:
            st.metric("Faithfulness", "N/A (no gate)")

    with col2:
        st.metric("Latency", f"{result['latency_sec']}s")

    with col3:
        st.metric("Chunks Retrieved", result["num_chunks_used"] if "num_chunks_used" in result else len(result["chunks_used"]))

    with col4:
        levels = result["chunk_levels"]
        level_str = f"{levels.count('city')}× city, {levels.count('state')}× state, {levels.count('federal')}× federal"
        st.metric("Source Levels", level_str)

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📄 Retrieved Evidence")

    chunk_ids = result["chunks_used"]
    chunk_levels = result["chunk_levels"]

    # Load chunk texts for display
    import json
    from pathlib import Path as P
    with open(P(__file__).parent.parent / "faiss_index" / "chunks_meta.json") as f:
        all_chunks = json.load(f)
    chunk_lookup = {c["chunk_id"]: c for c in all_chunks}

    level_colors = {"city": "🟦", "state": "🟩", "federal": "🟥"}

    for i, (cid, level) in enumerate(zip(chunk_ids, chunk_levels)):
        chunk = chunk_lookup.get(cid, {})
        icon = level_colors.get(level, "⬜")
        with st.expander(f"{icon} [{level.upper()}] {chunk.get('doc_name', cid)} — chunk {i+1}"):
            st.caption(f"chunk_id: `{cid}` | source: `{chunk.get('source_file', '')}`")
            st.markdown(chunk.get("text", "")[:800] + ("..." if len(chunk.get("text","")) > 800 else ""))

    # ── Faithfulness sentence breakdown ───────────────────────────────────────
    if result.get("sentence_scores"):
        st.divider()
        st.markdown("#### 🔬 Faithfulness Sentence Analysis")
        st.caption("Each sentence in the answer is checked for support in the retrieved chunks.")

        for s in result["sentence_scores"]:
            supported = s["is_supported"]
            icon = "✅" if supported else "❌"
            score = s["entailment_score"]
            st.markdown(f"{icon} `[{score:.2f}]` {s['sentence']}")

elif run_btn and not query.strip():
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data sources: NYC Local Law 97 · OneNYC 2050 · PlaNYC · NY CLCPA Scoping Plan · Paris Agreement · Inflation Reduction Act · and more.")