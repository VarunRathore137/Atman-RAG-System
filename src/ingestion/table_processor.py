from typing import List, Optional
from src.ingestion.models import ExtractedTable

class TableProcessor:
    @staticmethod
    def clean_cell(cell: Optional[str]) -> str:
        if cell is None:
            return ""
        cleaned = str(cell).replace("\n", " ").strip()
        cleaned = cleaned.replace("|", "/")
        return cleaned

    @classmethod
    def to_markdown(
        cls,
        table_rows: List[List[Optional[str]]],
        table_id: str = "table",
        page_num: int = 1
    ) -> Optional[ExtractedTable]:
        if not table_rows or len(table_rows) < 1:
            return None
        cleaned_rows = [[cls.clean_cell(cell) for cell in row] for row in table_rows]
        valid_rows = [row for row in cleaned_rows if any(cell.strip() for cell in row)]
        if not valid_rows:
            return None
        max_cols = max(len(row) for row in valid_rows)
        if max_cols == 0:
            return None
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in valid_rows]
        headers = normalized_rows[0]
        if not any(h.strip() for h in headers):
            headers = [f"Col_{i+1}" for i in range(max_cols)]
            data_rows = normalized_rows
        else:
            headers = [h.strip() if h.strip() else f"Col_{i+1}" for i, h in enumerate(headers)]
            data_rows = normalized_rows[1:]
        header_line ="| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * max_cols) + " |"
        data_lines = ["| " + " | ".join(row) + " |" for row in data_rows if any(cell.strip() for cell in row)]
        markdown_lines = [header_line, separator_line] + data_lines
        markdown_content = "\n".join(markdown_lines)
        return ExtractedTable(
            table_id=table_id,
            page_num=page_num,
            markdown_content=markdown_content,
            row_count=len(data_rows) + 1,
            col_count=max_cols,
            raw_headers=headers,
        )
