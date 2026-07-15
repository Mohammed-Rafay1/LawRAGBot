"""
Law Guide: RAG Evaluation Scorecard
Runs baseline query tests to measure category accuracy, off-topic guardrails, and response relevance.

Usage: python src/eval.py
"""

import sys
import os
import asyncio

# Ensure src is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import run_query

TEST_CASES = [
    {
        "query": "What is the punishment for murder under Section 302 PPC?",
        "expected_category": "Criminal",
        "expected_keywords": ["death", "imprisonment", "life", "penal", "fine"],
        "is_legal": True
    },
    {
        "query": "How can a wife file for Khula in the family court of Pakistan?",
        "expected_category": "Family",
        "expected_keywords": ["khula", "dissolution", "dower", "marriage", "family"],
        "is_legal": True
    },
    {
        "query": "What is the specific performance of a contract under the Specific Relief Act?",
        "expected_category": "Civil",
        "expected_keywords": ["performance", "specific", "relief", "contract", "agreement"],
        "is_legal": True
    },
    {
        "query": "What are the rules for registering a company under the Companies Act 2017?",
        "expected_category": "Corporate",
        "expected_keywords": ["companies", "act", "director", "secp", "incorporation"],
        "is_legal": True
    },
    {
        "query": "What is the definition of fundamental rights under Article 9 of the Constitution of Pakistan?",
        "expected_category": "Constitutional",
        "expected_keywords": ["life", "liberty", "security", "fundamental", "rights"],
        "is_legal": True
    },
    {
        "query": "What is the tax rate for salaried individuals in Finance Act 2025?",
        "expected_category": "Tax",
        "expected_keywords": ["tax", "rate", "income", "salary", "salaried", "finance"],
        "is_legal": True
    },
    {
        "query": "Can you give me recipes for baking a chocolate cake at home?",
        "expected_category": "Off-Topic",
        "expected_keywords": ["off-topic", "cannot", "legal", "law", "pakistani"],
        "is_legal": False
    },
    {
        "query": "Write a python script to reverse a linked list.",
        "expected_category": "Off-Topic",
        "expected_keywords": ["off-topic", "cannot", "legal", "law", "pakistani"],
        "is_legal": False
    }
]

async def run_evaluation():
    print("=" * 70)
    print("⚖️  LAW GUIDE: RAG Pipeline Evaluation Scorecard")
    print("=" * 70)

    total_tests = len(TEST_CASES)
    category_matches = 0
    keyword_matches = 0
    guardrail_matches = 0
    passed_tests = []
    failed_tests = []

    for idx, tc in enumerate(TEST_CASES, 1):
        query = tc["query"]
        expected_cat = tc["expected_category"]
        expected_kws = tc["expected_keywords"]
        is_legal = tc["is_legal"]

        print(f"\n[{idx}/{total_tests}] Evaluating Query: '{query}'")
        try:
            res = await run_query(query)
            actual_cat = res.get("category", "General")
            response = res.get("response", "").lower()
            is_draft = res.get("is_draft", False)
            confidence = res.get("confidence", "Low")

            # Check 1: Category Classification Accuracy
            cat_ok = actual_cat.lower() == expected_cat.lower()
            if cat_ok:
                category_matches += 1

            # Check 2: Guardrail Rejection Accuracy (Off-Topic check)
            if is_legal:
                guardrail_ok = actual_cat.lower() != "off-topic"
            else:
                guardrail_ok = actual_cat.lower() == "off-topic"
            
            if guardrail_ok:
                guardrail_matches += 1

            # Check 3: Content Relevance/Groundedness (keyword inclusion check)
            found_kws = [kw for kw in expected_kws if kw.lower() in response]
            kw_ratio = len(found_kws) / len(expected_kws)
            kw_ok = kw_ratio >= 0.40  # expect at least 40% keyword representation
            if kw_ok:
                keyword_matches += 1

            test_ok = cat_ok and guardrail_ok and kw_ok
            if test_ok:
                passed_tests.append(query)
                status = "✅ PASS"
            else:
                failed_tests.append(query)
                status = "❌ FAIL"

            print(f"   Status     : {status}")
            print(f"   Category   : Actual={actual_cat:<12} (Expected={expected_cat}) -> {'OK' if cat_ok else 'FAIL'}")
            print(f"   Guardrails : Intent Allowed={is_legal:<5} (Classifier Got Off-Topic={not guardrail_ok}) -> {'OK' if guardrail_ok else 'FAIL'}")
            print(f"   Relevance  : Found {len(found_kws)}/{len(expected_kws)} expected keywords -> {'OK' if kw_ok else 'FAIL'}")
            print(f"   Confidence : {confidence}")

        except Exception as e:
            failed_tests.append(query)
            print(f"   ❌ Error running query: {e}")

    # Print Summary Report
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Category Accuracy : {category_matches}/{total_tests} ({category_matches/total_tests:.1%})")
    print(f"Guardrail Accuracy: {guardrail_matches}/{total_tests} ({guardrail_matches/total_tests:.1%})")
    print(f"Keyword Relevance : {keyword_matches}/{total_tests} ({keyword_matches/total_tests:.1%})")
    print(f"Overall Success   : {len(passed_tests)}/{total_tests} ({len(passed_tests)/total_tests:.1%})")
    print("=" * 70)

if __name__ == "__main__":
    # Force UTF-8 stdout encoding to support emojis in terminal output
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    asyncio.run(run_evaluation())
