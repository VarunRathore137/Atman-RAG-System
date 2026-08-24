"""
Phase 7 Evaluation benchmark tests:
Tests dataset integrity, benchmark evaluator metrics calculation, report generation, and system accuracy.
"""

import os
import pytest

from src.evaluation import (
    BENCHMARK_DATASET,
    BenchmarkEvaluator,
    generate_markdown_report,
)
from src.generation import RAGPipeline, MockLLMClient


def test_benchmark_dataset_integrity():
    """
    Test that the benchmark dataset contains 15 valid, balanced test cases.
    """
    assert len(BENCHMARK_DATASET) == 15

    categories = {case.category for case in BENCHMARK_DATASET}
    assert "direct_factual" in categories
    assert "table_reasoning" in categories
    assert "cross_doc" in categories
    assert "out_of_domain" in categories

    ood_cases = [c for c in BENCHMARK_DATASET if c.should_abstain]
    in_domain_cases = [c for c in BENCHMARK_DATASET if not c.should_abstain]

    assert len(ood_cases) == 4
    assert len(in_domain_cases) == 11


def test_benchmark_evaluator_execution(tmp_path):
    """
    Execute full 15-question benchmark with deterministic MockLLM and verify accuracy thresholds.
    """
    pipeline = RAGPipeline(llm_client=MockLLMClient())
    evaluator = BenchmarkEvaluator(pipeline=pipeline)

    log_path = str(tmp_path / "test_eval_log.json")
    summary = evaluator.run_benchmark(save_log_path=log_path)

    assert summary.total_queries == 15
    assert summary.in_domain_count == 11
    assert summary.out_of_domain_count == 4

    # Verification assertions
    assert summary.retrieval_recall_rate >= 0.85
    assert summary.abstention_f1 >= 0.90
    assert summary.mean_latency_ms > 0.0

    # Ensure log file was written
    assert os.path.exists(log_path)


def test_markdown_report_generation(tmp_path):
    """
    Test markdown report generator formats evaluation summary into GFM tables.
    """
    pipeline = RAGPipeline(llm_client=MockLLMClient())
    evaluator = BenchmarkEvaluator(pipeline=pipeline)
    summary = evaluator.run_benchmark(save_log_path=None)

    report_path = str(tmp_path / "test_report.md")
    report_text = generate_markdown_report(summary, output_path=report_path)

    assert "Atman Cloud Enterprise RAG — System Evaluation Report" in report_text
    assert "Executive Performance Summary" in report_text
    assert "Retrieval Recall@K" in report_text
    assert "Category-Level Performance Breakdown" in report_text
    assert os.path.exists(report_path)
