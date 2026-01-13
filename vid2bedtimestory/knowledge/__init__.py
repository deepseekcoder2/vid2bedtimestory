"""
Knowledge Base Module

Provides character databases for known franchises.
Supports:
- Package-included franchise data (ships with vid2bedtimestory)
- User-defined franchise data (~/.vid2bedtimestory/franchises/)

IMPORTANT: Franchise data is REQUIRED for the pipeline.
The system will refuse to operate without valid franchise JSON.
"""

from .loader import (
    load_franchise,
    list_available_franchises,
    validate_franchise,
    FranchiseData,
    CharacterData,
    PromptExamples,
    StyleRules,
    FranchiseValidationError,
)

__all__ = [
    "load_franchise",
    "list_available_franchises",
    "validate_franchise",
    "FranchiseData",
    "CharacterData",
    "PromptExamples",
    "StyleRules",
    "FranchiseValidationError",
]

