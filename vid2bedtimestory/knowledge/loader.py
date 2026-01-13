"""
Franchise Data Loader

Loads character databases with tiered resolution:
1. User config (~/.vid2bedtimestory/franchises/) - highest priority
2. Package data (vid2bedtimestory/knowledge/franchises/) - fallback

IMPORTANT: Franchise data is REQUIRED for the pipeline to run.
The system will refuse to operate without a valid franchise JSON.
"""

import json
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional


class FranchiseValidationError(Exception):
    """Raised when franchise data is missing or incomplete."""
    pass


@dataclass
class CharacterData:
    """
    Character information from franchise database.
    """
    id: str
    display_name: str
    pronoun: str
    role: str  # protagonist, ally, mentor, antagonist, neutral
    
    # Aliases for fuzzy matching
    aliases: list[str] = field(default_factory=list)
    
    # Visual signature for VLM matching
    visual_signature: dict = field(default_factory=dict)
    
    # Personality for story generation
    traits: list[str] = field(default_factory=list)
    catchphrases: list[str] = field(default_factory=list)
    speech_style: str = ""
    
    # Relationships (character_id -> relationship type)
    relationships: dict[str, str] = field(default_factory=dict)


@dataclass
class VisualStyleData:
    """Visual style guidance for screenshot selection."""
    shot_preferences: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    key_visual_elements: list[str] = field(default_factory=list)
    action_to_character_ratio: str = ""


@dataclass
class PaginationData:
    """Pagination settings for this franchise."""
    target_pages: int = 22
    min_pages: int = 18
    max_pages: int = 40
    words_per_page_target: int = 45
    sentences_per_paragraph_max: int = 3


@dataclass
class PromptExamples:
    """
    Franchise-specific examples for prompt injection.
    
    These replace hardcoded examples in prompts, making the system
    work with any franchise.
    """
    # Video analysis examples
    visual_description_good: str = ""
    visual_description_bad: str = ""
    frame_description_good: str = ""
    frame_description_bad: str = ""
    character_appearance_example: str = ""
    
    # Story writing examples
    story_opening: str = ""
    story_setup: str = ""
    story_problem: str = ""
    story_climax: str = ""
    story_resolution: str = ""
    
    # Pagination moment matching examples (list of dicts)
    pagination_examples: list[dict] = field(default_factory=list)
    
    # Visual target examples (list of dicts with page_text, visual_target)
    visual_target_examples: list[dict] = field(default_factory=list)
    
    # Deep dive context examples
    environment_examples: list[str] = field(default_factory=list)
    motion_examples: list[str] = field(default_factory=list)
    color_examples: list[str] = field(default_factory=list)
    sound_examples: list[str] = field(default_factory=list)
    
    # Beat detection examples (list of dicts)
    beat_examples: list[dict] = field(default_factory=list)


@dataclass
class StyleRules:
    """
    Franchise-specific style rules for story generation.
    """
    # Bold usage rules
    bold_rule: str = ""
    bold_examples: list[str] = field(default_factory=list)
    bold_never_for: list[str] = field(default_factory=list)
    
    # Paragraph style
    sentences_per_paragraph: str = ""
    dialogue_pattern: str = ""
    ending_exclamations: list[str] = field(default_factory=list)
    
    # Vocabulary
    preferred_words: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)
    
    # Pronoun example
    pronoun_wrong: str = ""
    pronoun_right: str = ""


@dataclass 
class FranchiseData:
    """
    Complete franchise database.
    
    REQUIRED for pipeline operation. Contains all franchise-specific
    content including prompt examples and style rules.
    """
    franchise_id: str
    franchise_name: str
    characters: dict[str, CharacterData]  # id -> CharacterData
    
    # Names that appear in subtitles but are NOT characters
    known_non_characters: list[str] = field(default_factory=list)
    
    # Generic metadata categories (e.g., vehicles, locations, items)
    metadata: dict[str, dict[str, dict]] = field(default_factory=dict)
    
    # Visual style guidance for screenshot selection
    visual_style: Optional[VisualStyleData] = None
    
    # Pagination settings
    pagination: Optional[PaginationData] = None
    
    # Scoring guidance for VLM
    scoring_guidance: str = ""
    
    # Prompt examples for injection into prompts
    prompt_examples: Optional[PromptExamples] = None
    
    # Style rules for story generation
    style_rules: Optional[StyleRules] = None
    
    # Metadata
    source: str = ""  # Where was this loaded from
    
    # Valid roles (extended to include villain and recurring)
    VALID_ROLES = {"protagonist", "ally", "mentor", "antagonist", "neutral", "villain", "recurring"}
    
    def get_all_catchphrases(self) -> list[str]:
        """Get all catchphrases from all characters."""
        phrases = []
        for char in self.characters.values():
            phrases.extend(char.catchphrases)
        return phrases
    
    @property
    def vehicles(self) -> dict[str, dict]:
        """Legacy access for vehicles (now part of metadata)."""
        return self.metadata.get("vehicles", {})
    
    def get_all_vehicle_names(self) -> list[str]:
        """Get all vehicle names."""
        return list(self.vehicles.keys())
    
    def get_character_names(self) -> list[str]:
        """Get all character display names."""
        return [c.display_name for c in self.characters.values()]
    
    def get_character(self, name: str) -> Optional[CharacterData]:
        """
        Look up character by name (exact, alias, or fuzzy match).
        """
        name_lower = name.lower().strip()
        
        # 1. Exact ID match
        if name_lower in self.characters:
            return self.characters[name_lower]
        
        # 2. Alias match
        for char_id, char in self.characters.items():
            if name_lower in [a.lower() for a in char.aliases]:
                return char
            if name_lower == char.display_name.lower():
                return char
        
        # 3. Fuzzy match (handles typos)
        all_names = list(self.characters.keys())
        for char in self.characters.values():
            all_names.extend([a.lower() for a in char.aliases])
            all_names.append(char.display_name.lower())
        
        matches = get_close_matches(name_lower, all_names, n=1, cutoff=0.8)
        if matches:
            # Find which character this matched
            matched_name = matches[0]
            for char_id, char in self.characters.items():
                if matched_name == char_id:
                    return char
                if matched_name == char.display_name.lower():
                    return char
                if matched_name in [a.lower() for a in char.aliases]:
                    return char
        
        return None
    
    def is_known_non_character(self, name: str) -> bool:
        """Check if name is in the exclusion list."""
        return name.lower().strip() in [n.lower() for n in self.known_non_characters]
    
    def get_visual_guidance_prompt(self) -> str:
        """
        Format visual_style into a prompt-friendly string for visual target generation.
        Returns empty string if no visual_style defined.
        """
        if not self.visual_style:
            return ""
        
        vs = self.visual_style
        lines = [f"FRANCHISE-SPECIFIC VISUAL GUIDANCE ({self.franchise_name}):"]
        
        if vs.shot_preferences:
            lines.append("PREFERRED SHOTS:")
            for pref in vs.shot_preferences:
                lines.append(f"  - {pref}")
        
        if vs.avoid:
            lines.append("AVOID THESE SHOTS:")
            for av in vs.avoid:
                lines.append(f"  - {av}")
        
        if vs.key_visual_elements:
            lines.append(f"KEY VISUAL ELEMENTS: {', '.join(vs.key_visual_elements)}")
        
        if vs.action_to_character_ratio:
            lines.append(f"SHOT BALANCE: {vs.action_to_character_ratio}")
        
        return "\n".join(lines)
    
    def get_scoring_guidance_prompt(self) -> str:
        """
        Format scoring_guidance into a prompt-friendly string for VLM scoring.
        Returns empty string if no scoring_guidance defined.
        """
        if not self.scoring_guidance:
            return ""
        return f"FRANCHISE CONTEXT: {self.scoring_guidance}"
    
    # =========================================================================
    # PROMPT INJECTION METHODS
    # =========================================================================
    
    def get_visual_description_examples(self) -> str:
        """
        Return formatted visual description examples for VIDEO_ANALYSIS prompts.
        """
        if not self.prompt_examples:
            raise FranchiseValidationError(
                f"Franchise '{self.franchise_id}' missing prompt_examples.video_analysis"
            )
        
        pe = self.prompt_examples
        return f"""BAD EXAMPLE: "{pe.visual_description_bad}"

GOOD EXAMPLE: "{pe.visual_description_good}" """
    
    def get_frame_description_examples(self) -> str:
        """
        Return formatted frame description examples for SPARSE_SURVEY prompts.
        """
        if not self.prompt_examples:
            raise FranchiseValidationError(
                f"Franchise '{self.franchise_id}' missing prompt_examples"
            )
        
        pe = self.prompt_examples
        return f"""GOOD EXAMPLE:
"{pe.frame_description_good}"

BAD EXAMPLE:
"{pe.frame_description_bad}" """
    
    def get_character_appearance_example(self) -> str:
        """Return character appearance example for prompts."""
        if not self.prompt_examples:
            return ""
        return self.prompt_examples.character_appearance_example
    
    def get_story_style_examples(self) -> str:
        """
        Return franchise-specific story style examples for STORY_WRITING prompts.
        """
        if not self.prompt_examples:
            raise FranchiseValidationError(
                f"Franchise '{self.franchise_id}' missing prompt_examples.story_writing"
            )
        
        pe = self.prompt_examples
        lines = [f"STYLE EXAMPLES ({self.franchise_name}):"]
        
        if pe.story_opening:
            lines.append(f'\nOPENING EXAMPLE (~45 words):\n"{pe.story_opening}"')
        if pe.story_setup:
            lines.append(f'\nSETUP EXAMPLE (~45 words):\n"{pe.story_setup}"')
        if pe.story_problem:
            lines.append(f'\nPROBLEM/ACTION EXAMPLE (~40 words):\n"{pe.story_problem}"')
        if pe.story_climax:
            lines.append(f'\nCLIMAX EXAMPLE (~50 words):\n"{pe.story_climax}"')
        if pe.story_resolution:
            lines.append(f'\nRESOLUTION EXAMPLE (~40 words):\n"{pe.story_resolution}"')
        
        return "\n".join(lines)
    
    def get_style_rules_prompt(self) -> str:
        """
        Return franchise-specific style rules for STORY_WRITING prompts.
        """
        if not self.style_rules:
            return ""
        
        sr = self.style_rules
        lines = [f"STYLE RULES ({self.franchise_name}):"]
        
        if sr.bold_rule:
            lines.append(f"\nBOLD TAG USAGE: {sr.bold_rule}")
            if sr.bold_examples:
                lines.append("Bold examples:")
                for ex in sr.bold_examples[:3]:
                    lines.append(f"  {ex}")
            if sr.bold_never_for:
                lines.append(f"Never use [bold] for: {', '.join(sr.bold_never_for)}")
        
        if sr.dialogue_pattern:
            lines.append(f"\nDIALOGUE PATTERN: {sr.dialogue_pattern}")
        
        if sr.ending_exclamations:
            lines.append(f"ENDING EXCLAMATIONS: {', '.join(sr.ending_exclamations)}")
        
        if sr.domain_terms:
            lines.append(f"\nDOMAIN VOCABULARY: {', '.join(sr.domain_terms)}")
        
        return "\n".join(lines)
    
    def get_pronoun_example(self) -> str:
        """Return pronoun usage example for prompts."""
        if not self.style_rules or not self.style_rules.pronoun_wrong:
            # Fallback to a generic example using first character with she/her
            for char in self.characters.values():
                if "she" in char.pronoun.lower():
                    return f'WRONG: "{char.display_name} waved his arms."\nRIGHT: "{char.display_name} waved her arms."'
            return ""
        
        sr = self.style_rules
        return f'WRONG: "{sr.pronoun_wrong}"\nRIGHT: "{sr.pronoun_right}"'
    
    def get_pagination_examples(self) -> str:
        """
        Return franchise-specific moment matching examples for PAGINATION prompts.
        """
        if not self.prompt_examples or not self.prompt_examples.pagination_examples:
            return ""
        
        lines = ["EXAMPLE MATCHING:"]
        for ex in self.prompt_examples.pagination_examples:
            lines.append(f'- Story: "{ex.get("story_text", "")}"')
            lines.append(f'  → Matches moment about "{ex.get("moment_description", "")}"')
        
        return "\n".join(lines)
    
    def get_visual_target_examples(self) -> str:
        """
        Return franchise-specific visual target examples for VISUAL_TARGETS prompts.
        """
        if not self.prompt_examples or not self.prompt_examples.visual_target_examples:
            return ""
        
        lines = ["EXAMPLES:"]
        for ex in self.prompt_examples.visual_target_examples:
            lines.append(f'\nPage text: "{ex.get("page_text", "")}"')
            lines.append(f'visual_target: "{ex.get("visual_target", "")}"')
        
        return "\n".join(lines)
    
    def get_deep_dive_examples(self) -> str:
        """
        Return franchise-specific examples for DEEP_DIVE prompts.
        """
        if not self.prompt_examples:
            return ""
        
        pe = self.prompt_examples
        lines = [f"FRANCHISE-SPECIFIC EXAMPLES ({self.franchise_name}):"]
        
        if pe.environment_examples:
            lines.append(f"Environment details: {', '.join(pe.environment_examples)}")
        if pe.motion_examples:
            lines.append(f"Motion/speed: {', '.join(pe.motion_examples)}")
        if pe.color_examples:
            lines.append(f"Colors/visuals: {', '.join(pe.color_examples)}")
        if pe.sound_examples:
            lines.append(f"Implied sounds: {', '.join(pe.sound_examples)}")
        
        return "\n".join(lines)
    
    def get_beat_examples(self) -> str:
        """
        Return franchise-specific beat detection examples.
        """
        if not self.prompt_examples or not self.prompt_examples.beat_examples:
            return ""
        
        lines = ["BEAT EXAMPLES:"]
        for ex in self.prompt_examples.beat_examples:
            lines.append(f"\n{ex.get('beat_type', 'unknown').upper()}:")
            lines.append(f"  Summary: {ex.get('summary', '')}")
            if ex.get('anchor_dialogue'):
                lines.append(f"  Key dialogue: {ex.get('anchor_dialogue')}")
        
        return "\n".join(lines)


def _get_user_franchises_dir() -> Path:
    """Get user's franchise config directory."""
    return Path.home() / ".vid2bedtimestory" / "franchises"


def _get_package_franchises_dir() -> Path:
    """Get package's franchise data directory."""
    return Path(__file__).parent / "franchises"


def load_franchise(franchise_id: str) -> Optional[FranchiseData]:
    """
    Load franchise database with tiered resolution.
    
    Priority:
    1. User config (~/.vid2bedtimestory/franchises/{id}.json)
    2. Package data (vid2bedtimestory/knowledge/franchises/{id}.json)
    
    Args:
        franchise_id: Franchise identifier (e.g., "hot_wheels_lets_race")
        
    Returns:
        FranchiseData or None if not found
    """
    franchise_id = franchise_id.lower().replace("-", "_").replace(" ", "_")
    filename = f"{franchise_id}.json"
    
    # Try user config first
    user_path = _get_user_franchises_dir() / filename
    if user_path.exists():
        return _load_franchise_file(user_path)
    
    # Fall back to package data
    package_path = _get_package_franchises_dir() / filename
    if package_path.exists():
        return _load_franchise_file(package_path)
    
    return None


def _load_franchise_file(path: Path) -> FranchiseData:
    """Load and parse a franchise JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Parse characters
    characters = {}
    for char_id, char_data in data.get("characters", {}).items():
        characters[char_id.lower()] = CharacterData(
            id=char_id.lower(),
            display_name=char_data.get("display_name", char_id),
            pronoun=char_data.get("pronoun", "they/them"),
            role=char_data.get("role", "neutral"),
            aliases=char_data.get("aliases", []),
            visual_signature=char_data.get("visual_signature", {}),
            traits=char_data.get("traits", []),
            catchphrases=char_data.get("catchphrases", []),
            speech_style=char_data.get("speech_style", ""),
            relationships=char_data.get("relationships", {}),
        )
    
    # Parse visual_style if present
    visual_style = None
    if "visual_style" in data:
        vs = data["visual_style"]
        visual_style = VisualStyleData(
            shot_preferences=vs.get("shot_preferences", []),
            avoid=vs.get("avoid", []),
            key_visual_elements=vs.get("key_visual_elements", []),
            action_to_character_ratio=vs.get("action_to_character_ratio", ""),
        )
    
    # Parse pagination if present
    pagination = None
    if "pagination" in data:
        pg = data["pagination"]
        pagination = PaginationData(
            target_pages=pg.get("target_pages", 22),
            min_pages=pg.get("min_pages", 18),
            max_pages=pg.get("max_pages", 40),
            words_per_page_target=pg.get("words_per_page_target", 45),
            sentences_per_paragraph_max=pg.get("sentences_per_paragraph_max", 3),
        )
    
    # Parse prompt_examples if present
    prompt_examples = None
    if "prompt_examples" in data:
        pe = data["prompt_examples"]
        va = pe.get("video_analysis", {})
        sw = pe.get("story_writing", {})
        dd = pe.get("deep_dive", {})
        
        prompt_examples = PromptExamples(
            # Video analysis
            visual_description_good=va.get("visual_description", {}).get("good", ""),
            visual_description_bad=va.get("visual_description", {}).get("bad", ""),
            frame_description_good=va.get("frame_description", {}).get("good", ""),
            frame_description_bad=va.get("frame_description", {}).get("bad", ""),
            character_appearance_example=va.get("character_appearance_example", ""),
            # Story writing
            story_opening=sw.get("opening", ""),
            story_setup=sw.get("setup", ""),
            story_problem=sw.get("problem", ""),
            story_climax=sw.get("climax", ""),
            story_resolution=sw.get("resolution", ""),
            # Pagination
            pagination_examples=pe.get("pagination", {}).get("moment_matching", []),
            # Visual targets
            visual_target_examples=pe.get("visual_targets", {}).get("examples", []),
            # Deep dive
            environment_examples=dd.get("environment_examples", []),
            motion_examples=dd.get("motion_examples", []),
            color_examples=dd.get("color_examples", []),
            sound_examples=dd.get("sound_examples", []),
            # Beat detection
            beat_examples=pe.get("beat_detection", {}).get("examples", []),
        )
    
    # Parse style_rules if present
    style_rules = None
    if "style_rules" in data:
        sr = data["style_rules"]
        bu = sr.get("bold_usage", {})
        ps = sr.get("paragraph_style", {})
        vocab = sr.get("vocabulary", {})
        pron = sr.get("pronoun_example", {})
        
        style_rules = StyleRules(
            bold_rule=bu.get("rule", ""),
            bold_examples=bu.get("examples", []),
            bold_never_for=bu.get("never_for", []),
            sentences_per_paragraph=ps.get("sentences_per_paragraph", ""),
            dialogue_pattern=ps.get("dialogue_pattern", ""),
            ending_exclamations=ps.get("ending_exclamations", []),
            preferred_words=vocab.get("preferred_words", []),
            domain_terms=vocab.get("domain_terms", []),
            pronoun_wrong=pron.get("wrong", ""),
            pronoun_right=pron.get("right", ""),
        )
    
    # Collect generic metadata (any top-level dict that isn't a known field)
    known_fields = {
        "franchise_id", "franchise_name", "characters", 
        "known_non_characters", "source", "visual_style", 
        "pagination", "scoring_guidance", "prompt_examples", "style_rules"
    }
    metadata = {}
    for key, value in data.items():
        if key not in known_fields and isinstance(value, dict):
            metadata[key] = value
            
    return FranchiseData(
        franchise_id=data.get("franchise_id", path.stem),
        franchise_name=data.get("franchise_name", path.stem),
        characters=characters,
        known_non_characters=data.get("known_non_characters", []),
        metadata=metadata,
        visual_style=visual_style,
        pagination=pagination,
        scoring_guidance=data.get("scoring_guidance", ""),
        prompt_examples=prompt_examples,
        style_rules=style_rules,
        source=str(path),
    )


def validate_franchise(franchise: FranchiseData) -> None:
    """
    Validate that a franchise has all required fields for the pipeline.
    
    Raises:
        FranchiseValidationError: If required fields are missing
    """
    missing = []
    
    # Required: characters
    if not franchise.characters:
        missing.append("characters (at least one)")
    
    # Required: prompt_examples
    if not franchise.prompt_examples:
        missing.append("prompt_examples section")
    else:
        pe = franchise.prompt_examples
        if not pe.visual_description_good:
            missing.append("prompt_examples.video_analysis.visual_description.good")
        if not pe.visual_description_bad:
            missing.append("prompt_examples.video_analysis.visual_description.bad")
        if not pe.story_opening:
            missing.append("prompt_examples.story_writing.opening")
        if not pe.story_climax:
            missing.append("prompt_examples.story_writing.climax")
    
    # Required: visual_style
    if not franchise.visual_style:
        missing.append("visual_style section")
    
    # Required: pagination
    if not franchise.pagination:
        missing.append("pagination section")
    
    if missing:
        raise FranchiseValidationError(
            f"Franchise '{franchise.franchise_id}' is incomplete.\n"
            f"Missing required fields:\n  - " + "\n  - ".join(missing) + "\n"
            f"See documentation for franchise JSON schema."
        )


def list_available_franchises() -> list[tuple[str, str, str]]:
    """
    List all available franchise databases.
    
    Returns:
        List of (franchise_id, franchise_name, source) tuples
    """
    franchises = []
    
    # Check package data
    package_dir = _get_package_franchises_dir()
    if package_dir.exists():
        for path in package_dir.glob("*.json"):
            try:
                data = _load_franchise_file(path)
                franchises.append((
                    data.franchise_id,
                    data.franchise_name,
                    "package",
                ))
            except Exception:
                continue
    
    # Check user config (may override package)
    user_dir = _get_user_franchises_dir()
    if user_dir.exists():
        for path in user_dir.glob("*.json"):
            try:
                data = _load_franchise_file(path)
                # Check if this overrides a package franchise
                existing = [f for f in franchises if f[0] == data.franchise_id]
                if existing:
                    franchises.remove(existing[0])
                franchises.append((
                    data.franchise_id,
                    data.franchise_name,
                    "user",
                ))
            except Exception:
                continue
    
    return sorted(franchises, key=lambda x: x[1])

