"""
Law Guide: LangGraph Pipeline
StateGraph: Router → Retriever → Grader → Generator / Drafter
- Vector DB    : Qdrant (local)
- Embeddings   : multi-qa-MiniLM-L6-cos-v1
- LLM          : Groq API
- Features     : Conversation memory, confidence score, document drafting, Urdu support
"""

import os
import re
from typing import TypedDict, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── Configuration ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_DIR      = os.path.join(BASE_DIR, "data", "qdrant_db")
COLLECTION_NAME = "pakistan_laws"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")   # set via: export GROQ_API_KEY=your_key
MAX_RETRIES     = 2


# ── State Schema ───────────────────────────────────────────────
class GraphState(TypedDict):
    query:         str
    chat_history:  List[str]       # conversation memory
    category:      str
    is_vague:      bool
    is_draft:      bool            # document drafting flag
    documents:     List[Document]
    relevance:     str
    retries:       int
    response:      str
    clarification: str
    confidence:    str             # High / Medium / Low
    error:         Optional[str]


# ── Shared Resources ───────────────────────────────────────────
_llm         = None
_router_llm  = None
_vectorstore = None
_bm25_index  = None
_all_documents = []
_cross_encoder = None


def get_llm():
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Run: export GROQ_API_KEY='your_key_here'\n"
                "Get a free key at: https://console.groq.com"
            )
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            api_key=GROQ_API_KEY,
        )
    return _llm


def get_router_llm():
    global _router_llm
    if _router_llm is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set.")
        _router_llm = ChatGroq(
            model="llama-3.1-8b-instant",  # faster & cheaper model for routing
            temperature=0.0,
            api_key=GROQ_API_KEY,
        )
    return _router_llm


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if qdrant_url:
            print("🌐 Connecting to Qdrant Cloud...", flush=True)
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            if not os.path.exists(QDRANT_DIR):
                print(f"❌ QDRANT LOCAL DIRECTORY NOT FOUND: {QDRANT_DIR}", flush=True)
                raise FileNotFoundError(
                    f"Qdrant store not found at: {QDRANT_DIR}\n"
                    "Run: python src/data.py  to build it first."
                )
            print("💾 Connecting to local Qdrant database...", flush=True)
            client = QdrantClient(path=QDRANT_DIR)

        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vectorstore


def get_bm25_index():
    global _bm25_index, _all_documents
    if _bm25_index is None:
        vs = get_vectorstore()
        client = vs.client
        
        print("⚡ Loading Qdrant points for BM25 indexing...")
        # Retrieve up to 30,000 points (our database is ~25,822)
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=30000,
            with_payload=True,
            with_vectors=False,
        )
        
        _all_documents = []
        tokenized_corpus = []
        
        for record in records:
            payload = record.payload
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            doc = Document(page_content=page_content, metadata=metadata)
            _all_documents.append(doc)
            
            # Simple whitespace tokenization for BM25 index
            tokens = page_content.lower().split()
            tokenized_corpus.append(tokens)
            
        _bm25_index = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 Index initialized with {len(_all_documents)} documents!")
        
    return _bm25_index, _all_documents


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("⚡ Loading Cross-Encoder: ms-marco-MiniLM-L-6-v2...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        print("✅ Cross-Encoder loaded!")
    return _cross_encoder


# ── Helper Functions ───────────────────────────────────────────
def _build_context(docs: List[Document]) -> str:
    if not docs:
        return "No relevant legal provisions found."
    return "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('law_name', 'Unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])


def _format_history(chat_history: List[str]) -> str:
    if not chat_history:
        return "No previous conversation."
    return "\n".join(chat_history[-10:])  # last 10 turns to avoid token overflow


# ── Node 1: Router ─────────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal query classifier for Pakistani law.
Analyze the query and respond with EXACTLY this format (nothing else):

CATEGORY: [one of: Criminal, Civil, Family, Corporate, Constitutional, Tax, Labour, Shariah, Off-Topic]
VAGUE: [YES or NO]
DRAFT: [YES or NO]

Rules:
- Criminal      : PPC offenses, FIR, bail, CrPC procedure, cybercrime, terrorism, drugs
- Civil         : Property, contracts, CPC procedure, specific performance, limitation
- Family        : Marriage, divorce, khula, inheritance, custody, Muslim Family Laws
- Corporate     : Company registration, shares, directors, partnership, negotiable instruments
- Constitutional: Fundamental rights, government structure, articles of constitution
- Tax           : Income tax, sales tax, customs, finance act
- Labour        : Employment, wages, factories, industrial relations, EOBI
- Shariah       : Hudood, qisas, diyat, Federal Shariat Court, Islamic law
- Off-Topic     : General greetings (hi, hello), general chit-chat, programming/coding, cooking, math, translation, general history, general knowledge, or any topic not directly related to Pakistani law/statutes.

CRITICAL GUARDRAIL RULE:
If the user's query is not directly related to legal advice, statutes, courts, or document drafting under the jurisdiction of Pakistan, you MUST classify it as CATEGORY: Off-Topic, VAGUE: NO, DRAFT: NO. Be strict.

CRITICAL FOLLOW-UP RULE:
If previous conversation exists, the new query is almost certainly a follow-up to that topic.
ALWAYS inherit the category from the prior conversation UNLESS the new query is explicitly about a completely different legal area.
Examples:
- Prior: salary dispute (Labour) → "Can I claim damages?" → CATEGORY: Labour
- Prior: property case (Civil) → "Can I get an injunction?" → CATEGORY: Civil
- Prior: divorce (Family) → "What about child custody?" → CATEGORY: Family
NEVER switch to Civil just because the word "damages" or "compensation" appears — check context first."""),
    ("human", "{query}")
])



async def router_node(state: GraphState) -> GraphState:
    llm = get_router_llm()

    # Build query with history context for better classification
    history_context    = _format_history(state.get("chat_history", []))
    query_with_context = state["query"]
    if history_context != "No previous conversation.":
        query_with_context = (
            f"[Previous conversation:\n{history_context}]\n\n"
            f"New query: {state['query']}"
        )

    chain = ROUTER_PROMPT | llm | StrOutputParser()
    try:
        result   = await chain.ainvoke({"query": query_with_context})
        lines    = result.strip().split("\n")
        category = "General"
        is_vague = False
        is_draft = False

        for line in lines:
            lu = line.upper().strip()
            if "CATEGORY:" in lu:
                cat = line.split(":", 1)[1].strip()
                for vc in ["Criminal", "Civil", "Family", "Corporate",
                           "Constitutional", "Tax", "Labour", "Shariah", "Off-Topic"]:
                    if vc.lower() in cat.lower():
                        category = vc
                        break
            elif "VAGUE:" in lu:
                is_vague = "YES" in lu
            elif "DRAFT:" in lu:
                is_draft = "YES" in lu

        # Robust fallbacks for smaller models like Llama 3.1 8B
        result_upper = result.upper()
        if "OFF-TOPIC" in result_upper or "OFF_TOPIC" in result_upper:
            category = "Off-Topic"

        # Override: never treat as vague if chat history exists
        if is_vague and state.get("chat_history"):
            is_vague = False

        clarification = ""
        if is_vague:
            clarify_chain = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a senior Pakistani lawyer. The client's query is too vague to advise on. "
                 "Ask ONE precise clarifying question a real lawyer would ask. "
                 "Be warm and professional. Reply in the client's language. "
                 "NEVER mix languages or scripts in your response."),
                ("human", "{query}")
            ]) | llm | StrOutputParser()
            clarification = await clarify_chain.ainvoke({"query": state["query"]})

        return {**state, "category": category, "is_vague": is_vague,
                "is_draft": is_draft, "clarification": clarification}

    except Exception as e:
        return {**state, "error": f"Router error: {str(e)}"}


# ── Node 2: Retriever ──────────────────────────────────────────
ROMAN_URDU_KEYWORDS = {
    "kaise": "how", "kya": "what", "milti": "granted",
    "hai": "is", "mein": "in", "ki saza": "punishment",
    "bail": "bail", "section": "section", "درخواست": "application",
    "talaq": "divorce", "khula": "khula", "waris": "inheritance",
    "muavza": "compensation", "nuksan": "damages", "giraftari": "arrest",
    "fard": "FIR complaint", "adalat": "court", "wakeel": "lawyer",
}


# ── Query Condensation Prompt & Node ──
CONDENSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal query re-writer for a conversational RAG system.
Given the conversation history and a follow-up query, re-write it into a single standalone, search-optimized legal question in English.
Make sure the re-written query is specific and contains all necessary legal context from the conversation history.
If the query is already standalone and does not refer to history, output the original query.
Do not add any explanation or preamble. Output ONLY the re-written query."""),
    ("human", "Conversation History:\n{chat_history}\n\nFollow-up Query: {query}\n\nStandalone Query:")
])


async def condense_query_node(state: GraphState) -> GraphState:
    history = state.get("chat_history", [])
    query = state["query"]

    if not history:
        return state

    router_llm = get_router_llm()
    chain = CONDENSATION_PROMPT | router_llm | StrOutputParser()
    try:
        formatted_history = _format_history(history)
        condensed_query = await chain.ainvoke({
            "chat_history": formatted_history,
            "query": query
        })
        condensed_query = condensed_query.strip()
        print(f"📝 Conversation Follow-Up: '{query}' -> Condensed Search Query: '{condensed_query}'")
        return {**state, "query": condensed_query}
    except Exception as e:
        print(f"⚠️ Query condensation error: {e}")
        return state


async def retriever_node(state: GraphState) -> GraphState:
    vs      = get_vectorstore()
    retries = state.get("retries", 0)
    query   = state["query"]

    # Expand Roman Urdu queries with English keywords for better retrieval
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    if not has_arabic:
        query_lower = query.lower()
        for urdu_word, eng_word in ROMAN_URDU_KEYWORDS.items():
            if urdu_word in query_lower:
                query = f"{query} {eng_word}"
                break

    try:
        # 1. Semantic Retrieval (Dense Vector Search)
        k_dense = 15 if retries > 0 else 10
        dense_docs = vs.similarity_search(query, k=k_dense)
        
        # 2. Lexical Retrieval (BM25 Keyword Search)
        bm25, all_docs = get_bm25_index()
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        k_sparse = 15 if retries > 0 else 10
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_sparse]
        sparse_docs = [all_docs[i] for i in top_indices if bm25_scores[i] > 0]
        
        # 3. Merge Candidate Pools (Union deduplicated by page_content + source_pdf)
        merged_candidates = []
        seen_contents = set()
        
        for doc in (dense_docs + sparse_docs):
            content_key = (doc.page_content.strip(), doc.metadata.get("source_pdf", ""))
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged_candidates.append(doc)
                
        # 4. Cross-Encoder Re-ranking
        if merged_candidates:
            cross_encoder = get_cross_encoder()
            pairs = [[query, doc.page_content] for doc in merged_candidates]
            scores = cross_encoder.predict(pairs)
            ranked_docs = [doc for _, doc in sorted(zip(scores, merged_candidates), key=lambda pair: pair[0], reverse=True)]
            final_docs = ranked_docs[:5]  # limit to top 5 chunks
        else:
            final_docs = []

        # Log final retrieved chunks to console
        print(f"\n============================================================")
        print(f"🔍 RETRIEVED CHUNKS ({len(final_docs)} re-ranked) for query: '{query}'")
        print(f"============================================================")
        for idx, doc in enumerate(final_docs, 1):
            source = doc.metadata.get('law_name', doc.metadata.get('source_pdf', 'Unknown Source'))
            page   = doc.metadata.get('page', '?')
            dtype  = doc.metadata.get('doc_type', 'statute').upper()
            print(f"\n[Chunk {idx}] | {dtype} | {source} (Page {page})")
            print("-" * 60)
            print(doc.page_content.strip())
            print("-" * 60)
        print(f"============================================================\n")

        return {**state, "documents": final_docs}
    except Exception as e:
        print(f"❌ RETRIEVAL EXCEPTION IN RETRIEVER NODE: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {**state, "documents": [], "error": f"Retrieval error: {str(e)}"}


async def off_topic_node(state: GraphState) -> GraphState:
    response = (
        "⚖️ **Law Guide Guardrail**\n\n"
        "I am an AI Legal Advisor dedicated exclusively to Pakistani law. "
        "Your query appears to be outside the scope of Pakistani legal matters, "
        "so I cannot provide an answer. Please ask a question related to statutes, "
        "family laws, corporate rules, or court judgments of Pakistan."
    )
    return {**state, "response": response, "confidence": "Low"}


# ── Node 3: Grader ─────────────────────────────────────────────
GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether retrieved legal chunks are relevant \
to a Pakistani law query. Respond with EXACTLY one word: RELEVANT or IRRELEVANT."""),
    ("human", "Query: {query}\n\nChunks:\n{context}\n\nVerdict:")
])


def grader_node(state: GraphState) -> GraphState:
    llm  = get_llm()
    docs = state.get("documents", [])

    if not docs:
        return {**state, "relevance": "irrelevant", "retries": state.get("retries", 0) + 1}

    try:
        result    = (GRADER_PROMPT | llm | StrOutputParser()).invoke(
            {"query": state["query"], "context": _build_context(docs)}
        )
        relevance = "relevant" if "RELEVANT" in result.upper().strip() else "irrelevant"
        retries   = state.get("retries", 0) + (1 if relevance == "irrelevant" else 0)
        return {**state, "relevance": relevance, "retries": retries}
    except Exception as e:
        return {**state, "relevance": "irrelevant",
                "retries": state.get("retries", 0) + 1,
                "error": f"Grader error: {str(e)}"}


# ── Node 4: Generator ──────────────────────────────────────────
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Pakistani advocate with 20 years of experience in Pakistani courts.
You speak with authority, precision, and empathy — like a real lawyer consulting a client.

STRICT RULES:
1. NEVER say "the provided chunks", "retrieved context", "source text", or "according to the database" — you are a lawyer, not an AI chatbot.
2. ALWAYS cite sections naturally:
   - "Under Section 302 of the PPC, your client faces..."
   - "The Supreme Court held in [Judgment Link/Name] that..."
   - "In your situation, Section 420 clearly applies because..."
3. Cite the exact source of your information using the document name and page number.
   Every chunk in the context below has a prefix like '[Source: Document Name | Page: Y]'.
   Whenever you mention a law, section, or court precedent, you MUST cite it inline using: `[Document Name, Page Y]`.
   Example: "...as stated in Section 302 of the PPC [Pakistan Penal Code 1860, Page 305]."
4. Enforce verbatim quotes: When quoting exact sections or court judgments, write the exact text in quotation marks `""` to verify grounding.
5. USE CASE HISTORY: If there is prior conversation, reference it naturally.
6. ADDRESS the user directly — use "you", "your case", "in your situation".
7. STRICT LAW GUARDRAIL: If the relevant legal provisions provided in the context below do not contain information related to the question, or are blank/empty, you MUST state: "I cannot find a specific legal provision or court precedent in my database to answer this query. As your legal advisor, I recommend consulting a practicing advocate for formal representation." Do NOT make up any laws.
8. STRUCTURE every response like a real legal consultation:
   - **Legal Position** — what the law says about their situation
   - **Applicable Sections** — exact citations with text and inline `[Document Name, Page Y]`
   - **Court Precedents** — relevant Supreme Court judgments (if available) with inline `[Document Name, Page Y]`
   - **Your Options** — what they can practically do
   - **Recommended Next Steps** — concrete action items
9. If criminal matter, always mention bailable vs non-bailable, warrant-less arrest, punishment, and bail.
10. If civil/family matter, mention jurisdiction, timeline, and out-of-court settlement possibilities.
11. LANGUAGE: Reply in the same language the user writes in.
    Urdu → Pure Urdu script only, NO Hindi, Devanagari, or Vietnamese mixed in.
    Roman Urdu → Roman Urdu | English → English
12. End naturally: "As your legal advisor, I strongly recommend consulting a practicing advocate for formal representation, as every case has unique facts that require personalized legal strategy."
"""),
    ("human", """Case History:
{chat_history}

Legal Question: {query}
Category: {category}

Relevant Legal Provisions & Judgments:
{context}

Please provide your legal guidance:""")
])



async def generator_node(state: GraphState) -> GraphState:
    llm  = get_llm()
    docs = state.get("documents", [])

    n          = len(docs)
    confidence = "High" if n >= 4 else "Medium" if n >= 2 else "Low"

    try:
        response = await (GENERATOR_PROMPT | llm | StrOutputParser()).ainvoke({
            "query":        state["query"],
            "category":     state.get("category", "General"),
            "context":      _build_context(docs),
            "chat_history": _format_history(state.get("chat_history", [])),
        })
        return {**state, "response": response, "confidence": confidence}
    except Exception as e:
        return {**state, "response": f"Error generating response: {str(e)}", "confidence": "Low"}


# ── Node 5: Drafter ────────────────────────────────────────────
DRAFTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Pakistani advocate drafting a formal legal document.

RULES:
1. Draft in proper legal format used in Pakistani courts.
2. Use formal legal language appropriate for Pakistani jurisdiction.
3. Include all standard sections for the requested document type.
4. Leave [PLACEHOLDER] for information the user must fill in
   (names, dates, amounts, addresses, CNIC numbers, case numbers etc.)
5. Cite the relevant law/section the document is based on.
6. Common documents you may draft:
   - Legal Notice (under Section 80 CPC or general)
   - Bail Application (under Section 497 CrPC)
   - Divorce / Talaq Notice (under Muslim Family Laws Ordinance 1961)
   - Khula Petition (under Family Courts Act 1964)
   - General Power of Attorney
   - Affidavit
   - Complaint / Application to police or magistrate
   - Tenancy Agreement
7. LANGUAGE: Draft in the same language the user requested.
8. End with: "Note: This is a draft template. Have it reviewed and signed by a licensed advocate before use."
"""),
    ("human", """Case History:
{chat_history}

Document Requested: {query}
Category: {category}

Relevant Legal Provisions:
{context}

Please draft the document:""")
])


async def drafter_node(state: GraphState) -> GraphState:
    llm  = get_llm()
    docs = state.get("documents", [])

    try:
        response = await (DRAFTER_PROMPT | llm | StrOutputParser()).ainvoke({
            "query":        state["query"],
            "category":     state.get("category", "General"),
            "context":      _build_context(docs),
            "chat_history": _format_history(state.get("chat_history", [])),
        })
        return {**state, "response": response, "confidence": "Medium"}
    except Exception as e:
        return {**state, "response": f"Error drafting document: {str(e)}", "confidence": "Low"}


# ── Fallback Node ──────────────────────────────────────────────
def fallback_node(state: GraphState) -> GraphState:
    return {**state, "confidence": "Low", "response": (
        "I cannot find a specific legal provision for this query in my current database.\n\n"
        "💡 **Tip:** Try rephrasing with specific section numbers or legal terms.\n\n"
        "As your legal advisor, I strongly recommend consulting a practicing advocate "
        "for formal representation, as every case has unique facts that require personalized legal strategy."
    )}


# ── Routing ────────────────────────────────────────────────────
def route_after_router(state: GraphState) -> str:
    if state.get("error"):           return "fallback"
    if state.get("is_vague", False): return END
    if state.get("category", "").lower() == "off-topic": return "off_topic"
    return "condense_query"


# ── Build Graph ────────────────────────────────────────────────
def build_graph():
    wf = StateGraph(GraphState)
    wf.add_node("router",            router_node)
    wf.add_node("condense_query",    condense_query_node)
    wf.add_node("retriever",         retriever_node)
    wf.add_node("generator",         generator_node)
    wf.add_node("drafter",           drafter_node)
    wf.add_node("off_topic",         off_topic_node)
    wf.add_node("fallback",          fallback_node)

    wf.set_entry_point("router")
    
    wf.add_conditional_edges("router", route_after_router, {
        "condense_query": "condense_query",
        "off_topic": "off_topic",
        "fallback": "fallback",
        END: END
    })
    
    wf.add_edge("condense_query", "retriever")
    wf.add_edge("off_topic", END)
    wf.add_edge("fallback", END)
    
    wf.add_conditional_edges("retriever", 
                             lambda s: "drafter" if s.get("is_draft") else "generator",
                             {"generator": "generator", "drafter": "drafter"})
                             
    wf.add_edge("generator", END)
    wf.add_edge("drafter",   END)
    return wf.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_query(query: str, chat_history: List[str] = None) -> dict:
    """
    Run a legal query through the graph.

    Args:
        query        : user's question
        chat_history : list of previous turns, e.g.:
                       ["User: my landlord...", "Assistant: Under Section..."]
    Returns:
        dict with keys: response, category, is_vague, is_draft, retries, confidence
    """
    result = await get_graph().ainvoke({
        "query":         query,
        "chat_history":  chat_history or [],
        "category":      "",
        "is_vague":      False,
        "is_draft":      False,
        "documents":     [],
        "relevance":     "",
        "retries":       0,
        "response":      "",
        "clarification": "",
        "confidence":    "Low",
        "error":         None,
    })

    if result.get("is_vague"):
        return {
            "response":   result.get("clarification", "Could you please provide more details?"),
            "category":   result.get("category", ""),
            "is_vague":   True,
            "is_draft":   False,
            "retries":    0,
            "confidence": "Low",
        }

    return {
        "response":   result.get("response", "An error occurred."),
        "category":   result.get("category", ""),
        "is_vague":   False,
        "is_draft":   result.get("is_draft", False),
        "retries":    result.get("retries", 0),
        "confidence": result.get("confidence", "Low"),
    }


# ── CLI Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    print("⚖️  Law Guide — Test Mode")
    print("=" * 50)

    history = []
    queries = [
        "My employer has not paid my salary for 3 months",
        "Can I also claim damages on top of that?",
        "Draft a legal notice to my employer for unpaid salary",
        "Bake me a cake please"
    ]

    async def main():
        for q in queries:
            print(f"\n📝 {q}")
            print("-" * 40)
            r = await run_query(q, chat_history=history)
            print(f"Category  : {r['category']}")
            print(f"Confidence: {r['confidence']}")
            print(f"Draft     : {r['is_draft']}")
            print(f"\n{r['response'][:600]}...")

            history.append(f"User: {q}")
            history.append(f"Assistant: {r['response'][:300]}")
            print("=" * 50)

    asyncio.run(main())