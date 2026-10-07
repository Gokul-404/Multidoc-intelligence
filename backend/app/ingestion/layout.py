"""
Layout detection and structure extraction for SentinelRAG.
Preserves headings, paragraphs, tables, lists, and figure captions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LayoutBlock:
    block_type: str  # heading | paragraph | table | list | figure_caption | other
    text: str
    page: int
    section: Optional[str] = None
    heading_level: Optional[int] = None  # 1-6 for headings


@dataclass
class PageLayout:
    page: int
    blocks: list[LayoutBlock] = field(default_factory=list)
    current_section: Optional[str] = None


@dataclass
class DocumentLayout:
    pages: list[PageLayout] = field(default_factory=list)
    title: Optional[str] = None
    detected_sections: list[str] = field(default_factory=list)


# Heading detection patterns (heuristic)
_HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$"),                      # Markdown headings
    re.compile(r"^(\d+\.(?:\d+\.)*)\s+([A-Z].{3,80})$"),   # Numbered headings: 1.2 Title
    re.compile(r"^([A-Z][A-Z\s]{4,60})$"),                  # ALL-CAPS headings
    re.compile(r"^(CHAPTER|SECTION|APPENDIX)\s+[\dIVXivx]+", re.IGNORECASE),
]

_TABLE_MARKERS = re.compile(
    r"(\|\s*[-–—]{2,}\s*\|)|"   # Markdown table separator
    r"(^\s*\+[-+]+\+\s*$)",      # ASCII table
    re.MULTILINE,
)

_LIST_MARKER = re.compile(r"^(\s*[-•*]\s+|\s*\d+[.)]\s+)", re.MULTILINE)

_FIGURE_CAPTION = re.compile(
    r"^(figure|fig|table|tab|exhibit|chart|diagram|appendix)[\s.:\d]",
    re.IGNORECASE,
)


def _detect_heading(line: str) -> Optional[tuple[int, str]]:
    """Return (heading_level, heading_text) or None."""
    for pattern in _HEADING_PATTERNS:
        m = pattern.match(line.strip())
        if m:
            if pattern.groups and len(m.groups()) >= 2:
                prefix = m.group(1)
                text = m.group(2)
                level = prefix.count("#") if "#" in prefix else 1
                return level, text.strip()
            else:
                return 1, line.strip()
    return None


def extract_layout(page_texts: dict[int, str]) -> DocumentLayout:
    """
    Convert raw per-page text into a structured DocumentLayout.
    page_texts: {page_number (1-indexed): raw_text}
    """
    layout = DocumentLayout()
    current_section: Optional[str] = None

    for page_num, raw_text in sorted(page_texts.items()):
        page_layout = PageLayout(page=page_num)
        lines = raw_text.split("\n")

        buffer: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                # Flush buffer as paragraph
                if buffer:
                    para_text = " ".join(buffer).strip()
                    if para_text:
                        page_layout.blocks.append(
                            LayoutBlock(
                                block_type="paragraph",
                                text=para_text,
                                page=page_num,
                                section=current_section,
                            )
                        )
                    buffer = []
                i += 1
                continue

            # Check for heading
            heading_result = _detect_heading(stripped)
            if heading_result:
                # Flush buffer first
                if buffer:
                    para_text = " ".join(buffer).strip()
                    if para_text:
                        page_layout.blocks.append(
                            LayoutBlock(
                                block_type="paragraph",
                                text=para_text,
                                page=page_num,
                                section=current_section,
                            )
                        )
                    buffer = []
                level, text = heading_result
                current_section = text
                page_layout.blocks.append(
                    LayoutBlock(
                        block_type="heading",
                        text=text,
                        page=page_num,
                        section=text,
                        heading_level=level,
                    )
                )
                if text not in layout.detected_sections:
                    layout.detected_sections.append(text)
                i += 1
                continue

            # Check for table block (collect multi-line)
            if _TABLE_MARKERS.search(stripped):
                if buffer:
                    para_text = " ".join(buffer).strip()
                    if para_text:
                        page_layout.blocks.append(
                            LayoutBlock(block_type="paragraph", text=para_text, page=page_num, section=current_section)
                        )
                    buffer = []
                table_lines = [line]
                i += 1
                while i < len(lines) and (lines[i].strip() or _TABLE_MARKERS.search(lines[i])):
                    table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines)
                page_layout.blocks.append(
                    LayoutBlock(block_type="table", text=table_text, page=page_num, section=current_section)
                )
                continue

            # Check for figure/caption
            if _FIGURE_CAPTION.match(stripped):
                if buffer:
                    page_layout.blocks.append(
                        LayoutBlock(block_type="paragraph", text=" ".join(buffer).strip(), page=page_num, section=current_section)
                    )
                    buffer = []
                page_layout.blocks.append(
                    LayoutBlock(block_type="figure_caption", text=stripped, page=page_num, section=current_section)
                )
                i += 1
                continue

            # Check for list item
            if _LIST_MARKER.match(line):
                buffer.append(stripped)
                i += 1
                continue

            # Default: accumulate as paragraph
            buffer.append(stripped)
            i += 1

        # Flush remaining buffer
        if buffer:
            para_text = " ".join(buffer).strip()
            if para_text:
                page_layout.blocks.append(
                    LayoutBlock(block_type="paragraph", text=para_text, page=page_num, section=current_section)
                )

        layout.pages.append(page_layout)

    # Detect title from first heading on first page
    if layout.pages and layout.pages[0].blocks:
        for block in layout.pages[0].blocks:
            if block.block_type == "heading":
                layout.title = block.text
                break

    return layout
