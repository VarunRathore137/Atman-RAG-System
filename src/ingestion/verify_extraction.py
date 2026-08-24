"""
Smoke test script for Phase 1: Verify all 7 PDFs extract correctly.
Run with:   uv run python -m src.ingestion.verify_extraction
"""
import sys
from collections import defaultdict
from pathlib import Path

from config import cfg, logger
from src.ingestion import PDFExtractor


EXPECTED_DOCS = {
    "API_Reference",
    "Employee_Handbook",
    "FAQ_Support",
    "Onboarding_Guide",
    "Pricing_and_SLA",
    "Product_Manual",
    "Security_Policy",
}


def run():
    print("\n" + "=" * 60)
    print(" Phase 1 Smoke Test: PDF Ingestion Verification")
    print("=" * 60 + "\n")

    extractor = PDFExtractor(doc_dir=cfg.DOCS_DIR)
    all_pages = extractor.extract_all()

    # Group by doc_name
    doc_map = defaultdict(list)
    for page in all_pages:
        doc_map[page.doc_name].append(page)

    # Print summary table
    hdr_fmt = "{:<25} {:>5} {:>6} {:<15} {:<12} {:<6}"
    row_fmt = "{:<25} {:>5} {:>6} {:<15} {:<12} {:<6}"
    print(hdr_fmt.format("DOC_NAME", "PAGES", "TABLES", "EXTRACTOR", "DOC_CODE", "VERSION"))
    print("-" * 80)

    for doc_name in sorted(doc_map.keys()):
        pages = doc_map[doc_name]
        total_tables = sum(len(p.tables) for p in pages)
        extractor_used = pages[0].extractor_used
        doc_code = pages[0].doc_code or "UNKNOWN"
        doc_version = pages[0].doc_version or "-"
        print(row_fmt.format(doc_name, len(pages), total_tables, extractor_used, doc_code, doc_version))

    print()

    # Show Pricing_and_SLA tables
    if "Pricing_and_SLA" in doc_map:
        print("== Pricing_and_SLA: Extracted Tables ==")
        for page in doc_map["Pricing_and_SLA"]:
            for table in page.tables:
                print(f"\n  [Table {table.table_id} | Page {table.page_num} |"
                      f" {table.row_count} rows x #{table.col_count} cols]")
                print(table.markdown_content)
        print()

    # Assertions
    errors = []

    found_docs = set(doc_map.keys())
    missing = EXPECTED_DOCS - found_docs
    if missing:
        errors.append(f"Missing docs: {missing}")

    if "Pricing_and_SLA" in doc_map:
        sla_pages = doc_map["Pricing_and_SLA"]
        all_sla_md = "\n".join(
            t.markdown_content for p in sla_pages for t in p.tables
        )
        if "Free" not in all_sla_md:
            errors.append("Pricing_and_SLA: 'Free' not found in any table")
    else:
        errors.append("Pricing_and_SLA not extracted at all")

    if errors:
        print("\nASSERTION FAILURES:")
        for err in errors:
            print(f"  X  {err}")
        sys.exit(1)
    else:
        print("ALL ASSERTIONS PASSED")
        print(f"Total pages extracted: {len(all_pages)} across {len(doc_map)} documents")


if __name__ == "__main__":
    run()
