"""
Law Guide: Automated Integration Tests for Production RAG Upgrades
Verifies query flow, hybrid retrieval, guardrails, and citation formatting.

Usage: python src/test_production_rag.py
"""

import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import run_query

async def test_legal_flow():
    print("🧪 TEST: Standard Legal Advice Query (Section 302 PPC)...")
    res = await run_query("What is the punishment for murder under Section 302 PPC?")
    assert res["is_vague"] is False, "Query should not be vague"
    assert res["category"] == "Criminal", f"Expected Criminal category, got {res['category']}"
    assert "death" in res["response"].lower() or "life" in res["response"].lower(), "Response missing key penalties"
    assert "[" in res["response"] and "Page" in res["response"], "Response missing inline source citations"
    print("✅ Legal Advice Flow test passed!")

async def test_off_topic_guardrail():
    print("🧪 TEST: Off-Topic Guardrail Reject (Cooking recipe)...")
    res = await run_query("Give me a recipe for chocolate cake.")
    assert res["category"] == "Off-Topic", f"Expected category Off-Topic, got {res['category']}"
    assert "Law Guide Guardrail" in res["response"], "Response missing guardrail warning"
    print("✅ Off-Topic Guardrail test passed!")

async def test_drafting_flow():
    print("🧪 TEST: Document Drafting Flow (Legal Notice)...")
    res = await run_query("Draft a legal notice to my landlord for refusing to return my security deposit.")
    assert res["is_draft"] is True, "Query should be classified as a drafting request"
    assert "[PLACEHOLDER]" in res["response"] or "placeholder" in res["response"].lower() or "[" in res["response"], "Draft missing placeholder tags"
    print("✅ Document Drafting Flow test passed!")

async def run_all_tests():
    print("=" * 60)
    print("🚀 Running Production RAG Automated Integration Tests...")
    print("=" * 60)
    
    try:
        await test_legal_flow()
        print("-" * 40)
        await test_off_topic_guardrail()
        print("-" * 40)
        await test_drafting_flow()
        print("=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except AssertionError as e:
        print(f"❌ Test Assertion Failure: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Test Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Force UTF-8 stdout encoding
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    asyncio.run(run_all_tests())
