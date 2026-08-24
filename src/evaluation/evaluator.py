import json
import statistics
import time
from typing import List, Optional

from config import logger
from src.evaluation.benchmark_dataset import BENCHMARK_DATASET
from src.evaluation.models import EvaluationCase, EvaluationResult, EvaluationSummary
from src.generation.pipeline import RAGPipeline
from src.generation.models import RAGResponse


class BenchmarkEvaluator:
    """
    Automated evaluation harness for evaluating RAGPipeline against benchmark queries.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline()

    def evaluate_case(self, case: EvaluationCase, response: RAGResponse) -> EvaluationResult:
        """
        Evaluate a single RAGResponse against the ground-truth EvaluationCase.
        """
        # 1. Abstention check
        if case.should_abstain:
            abstention_correct = response.is_abstained
        else:
            abstention_correct = not response.is_abstained

        # 2. Retrieval recall check
        if case.should_abstain:
            recall_pass = True
        else:
            retrieved_doc_names = set()
            if response.retrieval_response:
                retrieved_doc_names = {
                    r.doc_name for r in response.retrieval_response.results
                }
            # Passes if ANY expected document was successfully retrieved
            recall_pass = bool(
                set(case.expected_docs).intersection(retrieved_doc_names)
            )

        # 3. Citation precision check
        if case.should_abstain:
            citation_pass = True
        else:
            if not response.citations:
                citation_pass = False
            else:
                # Check if citations match expected docs
                cited_docs = {c.doc_name for c in response.citations}
                citation_pass = bool(
                    set(case.expected_docs).intersection(cited_docs)
                )

        # 4. Fact keyword matching score
        if case.should_abstain:
            fact_score = 1.0 if response.is_abstained else 0.0
        else:
            if not case.expected_facts:
                fact_score = 1.0
            else:
                answer_lower = response.answer.lower()
                matches = sum(
                    1 for fact in case.expected_facts if fact.lower() in answer_lower
                )
                fact_score = matches / len(case.expected_facts)

        return EvaluationResult(
            case=case,
            rag_response=response,
            retrieval_recall_pass=recall_pass,
            citation_precision_pass=citation_pass,
            abstention_correct=abstention_correct,
            fact_match_score=fact_score,
            latency_ms=response.total_latency_ms,
        )

    def run_benchmark(
        self,
        dataset: Optional[List[EvaluationCase]] = None,
        save_log_path: Optional[str] = "eval_log.json",
    ) -> EvaluationSummary:
        """
        Run the complete benchmark dataset through the RAG pipeline and aggregate performance metrics.
        """
        cases = dataset or BENCHMARK_DATASET
        results: List[EvaluationResult] = []

        logger.info(f"Starting Benchmark Evaluation: {len(cases)} test cases...")

        for i, case in enumerate(cases, 1):
            logger.info(
                f"[{i}/{len(cases)}] Evaluating '{case.id}' ({case.category}): '{case.query}'"
            )
            resp = self.pipeline.query(case.query)
            res = self.evaluate_case(case, resp)
            results.append(res)
            # Throttle slightly to stay within API rate limits during bulk evaluation
            if i < len(cases):
                time.sleep(2.0)

        # Compute summary statistics
        in_domain_results = [r for r in results if not r.case.should_abstain]
        ood_results = [r for r in results if r.case.should_abstain]

        in_domain_count = len(in_domain_results)
        ood_count = len(ood_results)

        # Recall on in-domain
        recall_passes = sum(1 for r in in_domain_results if r.retrieval_recall_pass)
        recall_rate = recall_passes / in_domain_count if in_domain_count else 1.0

        # Citation precision on in-domain
        citation_passes = sum(1 for r in in_domain_results if r.citation_precision_pass)
        citation_rate = citation_passes / in_domain_count if in_domain_count else 1.0

        # Fact match rate on in-domain
        avg_fact_match = (
            statistics.mean([r.fact_match_score for r in in_domain_results])
            if in_domain_results
            else 1.0
        )

        # Abstention precision, recall, F1
        tp = sum(1 for r in ood_results if r.rag_response.is_abstained)  # True OOD abstained
        fp = sum(1 for r in in_domain_results if r.rag_response.is_abstained)  # In-domain falsely abstained
        fn = sum(1 for r in ood_results if not r.rag_response.is_abstained)  # OOD failed to abstain

        abstention_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        abstention_rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        if abstention_prec + abstention_rec > 0:
            abstention_f1 = (
                2 * (abstention_prec * abstention_rec) / (abstention_prec + abstention_rec)
            )
        else:
            abstention_f1 = 0.0

        # Latencies
        latencies = [r.latency_ms for r in results]
        latencies_sorted = sorted(latencies)
        mean_lat = statistics.mean(latencies)
        p50_lat = statistics.median(latencies)
        p95_lat = latencies_sorted[int(0.95 * len(latencies_sorted))]
        min_lat = min(latencies)
        max_lat = max(latencies)

        summary = EvaluationSummary(
            total_queries=len(cases),
            in_domain_count=in_domain_count,
            out_of_domain_count=ood_count,
            retrieval_recall_rate=round(recall_rate, 4),
            citation_precision_rate=round(citation_rate, 4),
            grounded_fact_match_rate=round(avg_fact_match, 4),
            abstention_precision=round(abstention_prec, 4),
            abstention_recall=round(abstention_rec, 4),
            abstention_f1=round(abstention_f1, 4),
            mean_latency_ms=round(mean_lat, 2),
            p50_latency_ms=round(p50_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            min_latency_ms=round(min_lat, 2),
            max_latency_ms=round(max_lat, 2),
            results=results,
        )

        logger.info(
            f"Benchmark Complete! Recall: {summary.retrieval_recall_rate*100:.1f}%, "
            f"Citation Precision: {summary.citation_precision_rate*100:.1f}%, "
            f"Abstention F1: {summary.abstention_f1:.3f}, "
            f"Mean Latency: {summary.mean_latency_ms:.1f}ms"
        )

        if save_log_path:
            self.save_json_log(summary, save_log_path)

        return summary

    def save_json_log(self, summary: EvaluationSummary, filepath: str = "eval_log.json"):
        """Save structured evaluation output to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        logger.info(f"Saved evaluation JSON log to: {filepath}")


if __name__ == "__main__":
    from src.evaluation.report_generator import (
        generate_markdown_report,
        generate_sample_qa_log,
    )
    evaluator = BenchmarkEvaluator()
    summary = evaluator.run_benchmark(save_log_path="eval_log.json")
    generate_markdown_report(summary, output_path="EVALUATION_REPORT.md")
    generate_sample_qa_log(summary, output_path="SAMPLE_QA_LOG.md")
    print(f"\nBenchmark Complete!")
    print(f"Recall Rate: {summary.retrieval_recall_rate * 100:.1f}%")
    print(f"Citation Precision: {summary.citation_precision_rate * 100:.1f}%")
    print(f"Abstention F1: {summary.abstention_f1:.3f}")
    print(f"Mean Latency: {summary.mean_latency_ms:.1f}ms")
    print(f"Reports saved to eval_log.json, EVALUATION_REPORT.md, and SAMPLE_QA_LOG.md")
