"""
Prompt templates for all video analysis phases.

All prompts are centralized here for:
- Easy editing and iteration
- Version control of prompt changes
- Consistency across phases

IMPORTANT: Some prompts contain {placeholders} for franchise-specific content.
The calling code MUST inject franchise data before using these prompts.
"""

# =============================================================================
# PHASE 1: SPARSE SURVEY
# =============================================================================

# NOTE: This prompt requires franchise injection. Use get_sparse_survey_prompt() instead.
SPARSE_SURVEY_PROMPT_TEMPLATE = """Describe this frame from a children's TV show in 2-3 sentences.

Include:
- Who is visible (character names if shown/spoken, otherwise describe appearance)
- What they are doing (specific actions, not vague descriptions)
- Where they are (location/environment details)
- Any text, signs, or UI elements visible

Be SPECIFIC about character appearances:
- Hair color and style
- Clothing colors and style
- Gender presentation (use he/she/they)
- Approximate age (child, teen, adult)

{franchise_frame_examples}

Output only the description, no JSON formatting needed."""


def get_sparse_survey_prompt(franchise_db) -> str:
    """Get SPARSE_SURVEY prompt with franchise examples injected."""
    if franchise_db and franchise_db.prompt_examples:
        examples = franchise_db.get_frame_description_examples()
    else:
        # This should not happen - franchise is required
        raise ValueError("Franchise database with prompt_examples is required")
    
    return SPARSE_SURVEY_PROMPT_TEMPLATE.format(franchise_frame_examples=examples)


# Legacy alias for backwards compatibility during transition
SPARSE_SURVEY_PROMPT = SPARSE_SURVEY_PROMPT_TEMPLATE


# =============================================================================
# PHASE 2: BEAT DETECTION
# =============================================================================

BEAT_DETECTION_SYSTEM = """You are a story structure analyst for children's television.

Your task: Given frame descriptions and dialogue from a video, identify the major story beats.

STORY BEAT TYPES:
- setup: Introduction of characters, setting, and initial situation
- inciting_incident: The ONE event that disrupts the status quo and kicks off the main conflict
- rising_action: Complications, obstacles, attempts to solve the problem
- climax: The peak moment of tension/conflict resolution
- resolution: Aftermath, lessons learned, return to normalcy

RULES:
1. Identify 6-10 beats that cover the ENTIRE video duration
2. Beats must NOT overlap - each timestamp belongs to exactly one beat
3. There should be exactly ONE inciting_incident and ONE climax
4. Use dialogue timestamps to anchor your time ranges
5. anchor_dialogue should contain 2-3 KEY lines that define this beat

OUTPUT FORMAT (strict JSON):
{
  "beats": [
    {
      "beat_id": "beat_01",
      "beat_type": "setup",
      "summary": "1-2 sentence description of what happens",
      "time_range": [start_seconds, end_seconds],
      "anchor_dialogue": ["key line 1", "key line 2"]
    }
  ]
}

Output ONLY valid JSON. No markdown, no explanations."""


BEAT_DETECTION_USER = """Analyze this children's TV episode for story structure.

VIDEO DURATION: {duration_s:.1f} seconds

FRAME DESCRIPTIONS (sampled every ~{sample_interval:.0f} seconds):
{captions_text}

DIALOGUE WITH TIMESTAMPS:
{subtitles_text}

{franchise_beat_examples}

Identify 6-10 story beats covering the full episode from 0 to {duration_s:.0f} seconds.

Remember:
- First beat should start near 0 seconds
- Last beat should end near {duration_s:.0f} seconds
- Use the dialogue timestamps to anchor your time ranges accurately"""


# =============================================================================
# PHASE 3: DEEP DIVE
# =============================================================================

# NOTE: This prompt requires franchise injection via {franchise_deep_dive_examples}
DEEP_DIVE_PROMPT = """Describe this frame in VIVID DETAIL for a children's book illustrator.

The illustrator will draw based ONLY on your description. Be extremely specific.

NEARBY DIALOGUE (for context):
{dialogue_context}

{character_reference}

Include ALL of these elements when present:

1. CHARACTER ACTIONS
   - What are they physically doing? Use specific verbs.
   - Examples: gripping, lunging, spinning, reaching, crouching

2. FACIAL EXPRESSIONS
   - What emotions show on their faces?
   - Eyes: wide, narrowed, squinting, teary
   - Mouth: grinning, gasping, grimacing, determined set

3. BODY LANGUAGE
   - Posture and gestures
   - Leaning forward, arms raised, shoulders hunched

4. ENVIRONMENT
   - Where exactly are they?
   - Be specific about setting details

5. MOTION & SPEED
   - Speed lines, blur effects, motion indicators
   - Direction of movement

6. COLORS & VISUAL EFFECTS
   - Key colors that stand out
   - Special effects or lighting

7. SOUNDS IMPLIED
   - What would we hear in this moment?

8. CAMERA ANGLE
   - Close-up, wide shot, low angle, bird's eye
   - This helps the illustrator frame the scene

{franchise_deep_dive_examples}

OUTPUT FORMAT (JSON):
{{
  "visual_description": "50-150 word vivid description covering all elements above",
  "emotional_beat": "single word for dominant emotion (excitement, fear, triumph, relief, joy, tension)"
}}

Output ONLY valid JSON."""


# =============================================================================
# PHASE 4: CHARACTER EXTRACTION
# =============================================================================

CHARACTER_PROMPT = """Describe the character "{name}" visible in this frame.

Focus on PHYSICAL APPEARANCE for a character reference sheet.

Provide:

1. role: Choose exactly ONE:
   - protagonist (main character driving the story)
   - ally (helps/supports the protagonist)
   - mentor (teaches, guides, or leads the protagonist)
   - antagonist (opposes or creates conflict for protagonist)
   - neutral (background character, minor role)

2. appearance: Detailed physical description including:
   - Gender presentation and appropriate pronoun
   - Hair: color, length, style (spiky, ponytail, short, etc.)
   - Skin tone
   - Clothing: colors, style, any logos or text
   - Age category: child (5-10), preteen (10-13), teen (13-18), adult (18+)
   - Distinctive features: glasses, hat, scars, accessories
   - Body type if notable: athletic, lanky, stocky

3. traits: 2-4 personality traits VISIBLE in this frame
   - Based on facial expression, posture, actions
   - Examples: confident, nervous, excited, mischievous, determined, kind

4. pronoun: The appropriate pronoun - "he", "she", or "they"

EXAMPLE OUTPUT FORMAT:
{{
  "role": "mentor|protagonist|ally|antagonist|neutral",
  "appearance": "Detailed physical description...",
  "traits": ["trait1", "trait2", "trait3"],
  "pronoun": "he|she|they"
}}

Output ONLY valid JSON."""


# =============================================================================
# PHASE 5: TITLE GENERATION (used in assembly)
# =============================================================================

TITLE_GENERATION_PROMPT = """Generate 3-5 engaging book title options for a children's picture book based on this story.

STORY BEATS:
{beats_summary}

MAIN CHARACTER: {protagonist_name}

REQUIREMENTS:
- Titles should be exciting and action-oriented
- Suitable for ages 5-8
- 3-6 words each
- Capture the main adventure/conflict

OUTPUT FORMAT (JSON):
{{
  "titles": ["Title One", "Title Two", "Title Three"]
}}

Output ONLY valid JSON."""


# =============================================================================
# PHASE 6: REFINEMENT (optional)
# =============================================================================

REFINEMENT_CHECK_PROMPT = """Review this video analysis for completeness.

ANALYSIS SUMMARY:
- Moments captured: {moment_count}
- Timeline coverage: {start_s:.0f}s to {end_s:.0f}s (video is {duration_s:.0f}s)
- Dialogue lines captured: {dialogue_captured} of {dialogue_total}
- Characters identified: {character_count}

UNCOVERED DIALOGUE (timestamps where no moment exists):
{uncovered_dialogue}

QUESTIONS:
1. Are there significant story events in the uncovered sections? (yes/no)
2. If yes, what specific frames should we capture? List timestamps.

OUTPUT FORMAT (JSON):
{{
  "has_gaps": true/false,
  "additional_timestamps": [123.4, 234.5],
  "gap_descriptions": ["Description of what's missing at ~123s", "..."]
}}

Output ONLY valid JSON."""

