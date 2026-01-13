from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import BookSpec


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class QualityValidator:
    """Validates book specifications against quality gates."""

    def __init__(
        self,
        min_pages: int = 18,
        max_pages: int = 28,
        min_chars_per_page: int = 50,
        max_chars_per_page: int = 800,
        max_repeated_paragraphs: int = 1,
        max_sentence_length: int = 25,  # words
        safety_blocklist: Optional[list[str]] = None,
    ):
        self.min_pages = min_pages
        self.max_pages = max_pages
        self.min_chars_per_page = min_chars_per_page
        self.max_chars_per_page = max_chars_per_page
        self.max_repeated_paragraphs = max_repeated_paragraphs
        self.max_sentence_length = max_sentence_length
        self.safety_blocklist = safety_blocklist or [
            "violence", "death", "kill", "hurt", "scary", "frighten",
            "blood", "weapon", "fight", "battle", "war"
        ]

    def validate_book_spec(
        self,
        book_spec: BookSpec,
        frames_dir: Optional[Path] = None
    ) -> list[str]:
        """Validate a book specification and return list of issues."""
        issues = []

        # QG1: Page count in allowed range
        page_count = len(book_spec.pages)
        if page_count < self.min_pages or page_count > self.max_pages:
            issues.append(
                f"QG1: Page count {page_count} not in range [{self.min_pages}, {self.max_pages}]"
            )

        # Check each page
        seen_texts = {}
        for page in book_spec.pages:
            page_num = page.page_index

            # QG2: No empty page text; no missing images
            if not page.text or not page.text.strip():
                issues.append(f"QG2: Page {page_num} has empty text")

            # Timestamp candidates should exist (even in fallback mode).
            if not page.image_timestamp_candidates_s:
                issues.append(f"QG2: Page {page_num} has no image timestamp candidates")

            # If frames are expected, the image file must exist.
            if frames_dir:
                frame_path = frames_dir / f"page_{page_num:03d}.png"
                if not frame_path.exists():
                    issues.append(f"QG2: Page {page_num} missing image file: {frame_path}")

            if page.text:
                text = page.text.strip()

                # QG3: Paragraph length bounds
                char_count = len(text)
                if char_count < self.min_chars_per_page:
                    issues.append(
                        f"QG3: Page {page_num} too short ({char_count} chars, min {self.min_chars_per_page})"
                    )
                elif char_count > self.max_chars_per_page:
                    issues.append(
                        f"QG3: Page {page_num} too long ({char_count} chars, max {self.max_chars_per_page})"
                    )

                # QG4: Detect repeated paragraphs
                text_key = text.lower().replace(" ", "").replace("\n", "")
                if text_key in seen_texts:
                    seen_texts[text_key] += 1
                    if seen_texts[text_key] > self.max_repeated_paragraphs:
                        issues.append(f"QG4: Page {page_num} has repeated content")
                else:
                    seen_texts[text_key] = 1

                # QG5: Basic readability heuristic
                readability_issues = self._check_readability(text, page_num)
                issues.extend(readability_issues)

                # QG6: Safety scan
                safety_issues = self._check_safety(text, page_num)
                issues.extend(safety_issues)

        return issues

    def _check_readability(self, text: str, page_num: int) -> list[str]:
        """Check basic readability heuristics."""
        issues = []

        # Split into sentences (basic heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            words = sentence.split()
            if len(words) > self.max_sentence_length:
                issues.append(
                    f"QG5: Page {page_num} sentence too long ({len(words)} words): '{sentence[:50]}...'"
                )

        # Check for uncommon words (very basic - words longer than 8 chars)
        words = re.findall(r'\b\w+\b', text.lower())
        long_words = [w for w in words if len(w) > 8 and not w.isdigit()]
        if len(long_words) > len(words) * 0.1:  # More than 10% long words
            issues.append(
                f"QG5: Page {page_num} may have complex vocabulary ({len(long_words)}/{len(words)} long words)"
            )

        return issues

    def _check_safety(self, text: str, page_num: int) -> list[str]:
        """Check for potentially inappropriate content using word boundaries."""
        issues = []
        text_lower = text.lower()

        for blocked_word in self.safety_blocklist:
            # Use word boundary regex to avoid false positives like "forward" matching "war"
            pattern = rf'\b{re.escape(blocked_word)}\b'
            if re.search(pattern, text_lower):
                issues.append(
                    f"QG6: Page {page_num} contains potentially unsafe content: '{blocked_word}'"
                )

        return issues

    def validate_or_raise(
        self,
        book_spec: BookSpec,
        frames_dir: Optional[Path] = None
    ) -> None:
        """Validate and raise ValidationError if issues found."""
        issues = self.validate_book_spec(book_spec, frames_dir)
        if issues:
            error_msg = f"Quality validation failed ({len(issues)} issues):\n" + "\n".join(f"  - {issue}" for issue in issues)
            raise ValidationError(error_msg)
