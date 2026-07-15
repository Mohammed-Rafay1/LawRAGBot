"""
Insaf-Guide: PDF Ingestion & Vector Store Builder
- Vector DB  : Qdrant (local, handles millions of vectors)
- Embeddings : multi-qa-MiniLM-L6-cos-v1 (faster + better for QA)
- Metadata   : law_category, law_name, year, court, doc_type

Usage: python src/data.py
"""

import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── Configuration ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR         = os.path.join(BASE_DIR, "data", "raw")
QDRANT_DIR      = os.path.join(BASE_DIR, "data", "qdrant_db")
COLLECTION_NAME = "pakistan_laws"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"   # faster + better for QA than all-MiniLM-L6-v2
VECTOR_SIZE     = 384                             # output dim of this model

# ── PDF Metadata Maps ──────────────────────────────────────────

PDF_CATEGORY_MAP = {
    # Constitutional
    "Constitution of Pakistan (1973).pdf"                                                          : "Constitutional",

    # Criminal
    "Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf"                    : "Criminal",
    "Code_of_Criminal_Procedure_1898_incorporating_amendments_to_16_February_2017.pdf"             : "Criminal",
    "The Prevention of Electronic Crimes Act, Rules Final Index ( Upto date 2025).pdf"             : "Criminal",
    "Qanun-e-Shahadat Order 1984.pdf"                                                              : "Criminal",
    "Anti-Terrorism Act 1997.pdf"                                                                  : "Criminal",
    "Control of Narcotic Substances Act 1997.pdf"                                                  : "Criminal",
    "Qisas and Diyat Ordinance 1990.pdf"                                                           : "Criminal",
    "Hudood Ordinances.pdf"                                                                        : "Criminal",

    # Civil
    "Code of Civil Procedure (CPC) 1908.pdf"                                                       : "Civil",
    "Contract Act 1872.pdf"                                                                        : "Civil",
    "Specific Relief Act 1877.pdf"                                                                 : "Civil",
    "Property-Law-Transfer-of-Property-Act-1872.pdf"                                               : "Civil",
    "2-limitation-act-1908-pdf.pdf"                                                                : "Civil",
    "Registration Act 1908.pdf"                                                                    : "Civil",

    # Family
    "Muslim Family Laws Ordinance 1961.pdf"                                                        : "Family",
    "Family Courts Act 1964.pdf"                                                                   : "Family",
    "Guardians and Wards Act 1890.pdf"                                                             : "Family",
    "Child Marriage Restraint Act 1929.pdf"                                                        : "Family",
    "Dowry and Bridal Gifts (Restriction) Act 1976.pdf"                                            : "Family",

    # Islamic / Shariah
    "Federal Shariat Court Act 1980.pdf"                                                           : "Shariah",
    "Enforcement of Shariah Act 1991.pdf"                                                          : "Shariah",

    # Corporate
    "companiesAct2017.pdf"                                                                         : "Corporate",
    "Partnership Act 1932.pdf"                                                                     : "Corporate",
    "Sales of Goods Act 1930.pdf"                                                                  : "Corporate",
    "Negotiable Instruments Act 1881.pdf"                                                          : "Corporate",

    # Tax
    "Income Tax Ordinance 2001.pdf"                                                                : "Tax",
    "Sales Tax Act 1990.pdf"                                                                       : "Tax",
    "Finance Act 2025.pdf"                                                                         : "Tax",
    "Customs Act 1969.pdf"                                                                         : "Tax",

    # Labour
    "Industrial Relations Act 2012.pdf"                                                            : "Labour",
    "Factories Act 1934.pdf"                                                                       : "Labour",
    "Payment of Wages Act 1936.pdf"                                                                : "Labour",
    "Minimum Wages Ordinance 1961.pdf"                                                             : "Labour",
}

PDF_LAW_NAMES = {
    "Constitution of Pakistan (1973).pdf"                                                          : "Constitution of Pakistan 1973",
    "Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf"                    : "Pakistan Penal Code 1860",
    "Code_of_Criminal_Procedure_1898_incorporating_amendments_to_16_February_2017.pdf"             : "Code of Criminal Procedure 1898",
    "The Prevention of Electronic Crimes Act, Rules Final Index ( Upto date 2025).pdf"             : "Prevention of Electronic Crimes Act 2025",
    "Qanun-e-Shahadat Order 1984.pdf"                                                              : "Qanun-e-Shahadat Order 1984",
    "Anti-Terrorism Act 1997.pdf"                                                                  : "Anti-Terrorism Act 1997",
    "Control of Narcotic Substances Act 1997.pdf"                                                  : "Control of Narcotic Substances Act 1997",
    "Qisas and Diyat Ordinance 1990.pdf"                                                           : "Qisas and Diyat Ordinance 1990",
    "Hudood Ordinances.pdf"                                                                        : "Hudood Ordinances",
    "Code of Civil Procedure (CPC) 1908.pdf"                                                       : "Code of Civil Procedure 1908",
    "Contract Act 1872.pdf"                                                                        : "Contract Act 1872",
    "Specific Relief Act 1877.pdf"                                                                 : "Specific Relief Act 1877",
    "Property-Law-Transfer-of-Property-Act-1872.pdf"                                               : "Transfer of Property Act 1882",
    "2-limitation-act-1908-pdf.pdf"                                                                : "Limitation Act 1908",
    "Registration Act 1908.pdf"                                                                    : "Registration Act 1908",
    "Muslim Family Laws Ordinance 1961.pdf"                                                        : "Muslim Family Laws Ordinance 1961",
    "Family Courts Act 1964.pdf"                                                                   : "Family Courts Act 1964",
    "Guardians and Wards Act 1890.pdf"                                                             : "Guardians and Wards Act 1890",
    "Child Marriage Restraint Act 1929.pdf"                                                        : "Child Marriage Restraint Act 1929",
    "Dowry and Bridal Gifts (Restriction) Act 1976.pdf"                                            : "Dowry and Bridal Gifts Act 1976",
    "Federal Shariat Court Act 1980.pdf"                                                           : "Federal Shariat Court Act 1980",
    "Enforcement of Shariah Act 1991.pdf"                                                          : "Enforcement of Shariah Act 1991",
    "companiesAct2017.pdf"                                                                         : "Companies Act 2017",
    "Partnership Act 1932.pdf"                                                                     : "Partnership Act 1932",
    "Sales of Goods Act 1930.pdf"                                                                  : "Sales of Goods Act 1930",
    "Negotiable Instruments Act 1881.pdf"                                                          : "Negotiable Instruments Act 1881",
    "Income Tax Ordinance 2001.pdf"                                                                : "Income Tax Ordinance 2001",
    "Sales Tax Act 1990.pdf"                                                                       : "Sales Tax Act 1990",
    "Finance Act 2025.pdf"                                                                         : "Finance Act 2025",
    "Customs Act 1969.pdf"                                                                         : "Customs Act 1969",
    "Industrial Relations Act 2012.pdf"                                                            : "Industrial Relations Act 2012",
    "Factories Act 1934.pdf"                                                                       : "Factories Act 1934",
    "Payment of Wages Act 1936.pdf"                                                                : "Payment of Wages Act 1936",
    "Minimum Wages Ordinance 1961.pdf"                                                             : "Minimum Wages Ordinance 1961",
}

# ── Court Judgment Filename Patterns ──────────────────────────
# Filenames follow pattern: Topic_Court_CaseNo_Year.pdf
# e.g. Murder_302PPC_crl.p._573_2025.pdf
# We extract year and court from the filename automatically

def parse_judgment_metadata(filename: str) -> dict:
    """
    Extract metadata from judgment PDF filename.
    Pattern: Topic_Court_CaseNo_Year.pdf
    Examples:
        Murder_302PPC_crl.p._573_2025.pdf
        Khula_c.p._308_p_2019.pdf
        Inheritance_c.a._184_2013.pdf
        Specific Performance_c.p._746_l_2025.pdf
        Dissolution of Muslim Marriages_c.p._308_p_2019.pdf
    """
    name = filename.replace(".pdf", "")
    parts = name.split("_")

    # Extract year — last numeric part
    year = "Unknown"
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4:
            year = part
            break

    # Extract topic — everything before the court code
    court_codes = ["c.p.", "c.a.", "crl.p.", "crl.a.", "j.p.", "s.m.c", "c.m.a"]
    topic = name
    for code in court_codes:
        if code in name.lower():
            topic = name[:name.lower().index(code)].strip("_").strip()
            break

    # Map court code to full name
    court = "Supreme Court of Pakistan"   # all your judgments are SC
    court_code_map = {
        "c.p."  : "Civil Petition",
        "c.a."  : "Civil Appeal",
        "crl.p.": "Criminal Petition",
        "crl.a.": "Criminal Appeal",
        "j.p."  : "Jail Petition",
        "c.m.a" : "Civil Misc Application",
    }
    petition_type = "Supreme Court Judgment"
    for code, label in court_code_map.items():
        if code in name.lower():
            petition_type = label
            break

    # Map topic to law category
    topic_category_map = {
        "murder"                        : "Criminal",
        "302"                           : "Criminal",
        "khula"                         : "Family",
        "dissolution"                   : "Family",
        "inheritance"                   : "Family",
        "specific performance"          : "Civil",
        "property"                      : "Civil",
        "contract"                      : "Civil",
        "bail"                          : "Criminal",
        "terrorism"                     : "Criminal",
        "constitutional"                : "Constitutional",
        "fundamental"                   : "Constitutional",
        "company"                       : "Corporate",
        "tax"                           : "Tax",
        "labour"                        : "Labour",
        "employment"                    : "Labour",
    }
    category = "General"
    topic_lower = topic.lower()
    for keyword, cat in topic_category_map.items():
        if keyword in topic_lower:
            category = cat
            break

    return {
        "law_category" : category,
        "law_name"     : f"{topic.replace('_', ' ').title()} — {petition_type}",
        "court"        : court,
        "year"         : year,
        "doc_type"     : "judgment",
        "petition_type": petition_type,
    }


def load_statute_pdfs():
    """Load all statute PDFs from PDF_DIR."""
    all_documents = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    if not pdf_files:
        print(f"⚠️  No statute PDFs found in: {PDF_DIR}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print(f"\n📚 Loading {len(pdf_files)} statute PDFs...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        category = PDF_CATEGORY_MAP.get(filename, "General")
        law_name = PDF_LAW_NAMES.get(filename, filename.replace(".pdf", ""))

        try:
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()

            total_chars = sum(len(p.page_content.strip()) for p in pages)
            if total_chars < 200:
                print(f"   ⚠️  {filename} — very little text ({total_chars} chars), may be scanned")
                continue

            for page in pages:
                page.metadata.update({
                    "source_pdf"  : filename,
                    "law_category": category,
                    "law_name"    : law_name,
                    "court"       : "N/A",
                    "year"        : extract_year_from_law_name(law_name),
                    "doc_type"    : "statute",
                })

            chunks = text_splitter.split_documents(pages)
            all_documents.extend(chunks)
            print(f"   ✅ {filename[:55]:<55} {len(pages):>4} pages → {len(chunks):>5} chunks")

        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")

    print(f"\n   Statute chunks total: {len(all_documents)}")
    return all_documents


def extract_year_from_law_name(law_name: str) -> str:
    """Extract year from law name like 'Companies Act 2017' → '2017'."""
    import re
    match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', law_name)
    return match.group(1) if match else "Unknown"


def load_judgment_pdfs(judgments_dir: str):
    """Load all court judgment PDFs from judgments_dir."""
    all_documents = []

    if not os.path.exists(judgments_dir):
        print(f"⚠️  Judgments directory not found: {judgments_dir}")
        print("   Skipping judgment PDFs.")
        return []

    pdf_files = glob.glob(os.path.join(judgments_dir, "*.pdf"))

    if not pdf_files:
        print(f"⚠️  No judgment PDFs found in: {judgments_dir}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    # slightly larger for judgments (more context needed)
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print(f"\n⚖️  Loading {len(pdf_files)} court judgment PDFs...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        metadata = parse_judgment_metadata(filename)

        try:
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()

            total_chars = sum(len(p.page_content.strip()) for p in pages)
            if total_chars < 200:
                print(f"   ⚠️  {filename} — very little text, skipping")
                continue

            for page in pages:
                page.metadata.update({
                    "source_pdf": filename,
                    **metadata,
                })

            chunks = text_splitter.split_documents(pages)
            all_documents.extend(chunks)
            print(f"   ✅ {filename[:55]:<55} {len(pages):>4} pages → {len(chunks):>5} chunks  [{metadata['year']}]")

        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")

    print(f"\n   Judgment chunks total: {len(all_documents)}")
    return all_documents


def create_qdrant_store(documents):
    """Create and persist Qdrant vector store."""

    print(f"\n🔧 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if qdrant_url:
        print("🌐 Connecting to Qdrant Cloud...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        # Remove existing DB
        if os.path.exists(QDRANT_DIR):
            shutil.rmtree(QDRANT_DIR)
            print(f"🗑️  Cleared old local Qdrant store")

        os.makedirs(QDRANT_DIR, exist_ok=True)
        print("💾 Creating local Qdrant database...")
        client = QdrantClient(path=QDRANT_DIR)

    # Recreate collection (safe delete and recreate)
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️  Cleared old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    # Instantiate the vector store using the existing client
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print(f"\n⏳ Embedding and uploading {len(documents)} chunks (batched)...")

    BATCH_SIZE = 500
    total      = len(documents)

    for i in range(0, total, BATCH_SIZE):
        batch     = documents[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        pct       = min(i + BATCH_SIZE, total)
        print(f"   Batch {batch_num}/{total_batches}  [{pct}/{total} chunks]...")

        vectorstore.add_documents(batch)

    print(f"\n✅ Qdrant store ready at: {QDRANT_DIR}")
    print(f"   Collection : {COLLECTION_NAME}")
    print(f"   Total vectors: {client.get_collection(COLLECTION_NAME).points_count}")
    return vectorstore


def main():
    print("=" * 60)
    print("⚖️  INSAF-GUIDE: Vector Store Builder (Qdrant)")
    print("=" * 60)

    # Load statutes
    statute_docs  = load_statute_pdfs()

    # Load judgments — put your judgment PDFs in data/judgments/
    judgments_path = os.path.join(BASE_DIR, "data", "judgments")
    judgment_docs  = load_judgment_pdfs(judgments_path)

    all_docs = statute_docs + judgment_docs

    if not all_docs:
        print("\n❌ No documents loaded. Check your PDF paths.")
        return

    print(f"\n📊 Total chunks: {len(statute_docs)} statutes + {len(judgment_docs)} judgments = {len(all_docs)}")

    create_qdrant_store(all_docs)

    print()
    print("=" * 60)
    print("✅ Done! Run: python src/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()