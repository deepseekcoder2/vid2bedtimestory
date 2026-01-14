"""
Centralized prompts for all LLM calls.

All prompts are stored here for easy editing and consistency.

IMPORTANT: These prompts contain {placeholders} for franchise-specific content.
The calling code MUST inject franchise data before using these prompts.
Hardcoded examples have been removed - all examples come from franchise JSON.
"""

# =============================================================================
# VIDEO ANALYSIS PROMPT (MLX-VLM / Qwen3-VL)
# =============================================================================

VIDEO_ANALYSIS_SYSTEM = """You are a children's media analyst extracting detailed visual data from video.

Your output will be used by a story writer who CANNOT see the video. Your descriptions are their ONLY source of visual information. Be extremely detailed and vivid.

OUTPUT SCHEMA (Strict JSON):
{
  "title_candidates": ["string - 3-5 engaging book title options"],
  "characters": [
    {
      "name": "string - as spoken in dialogue",
      "role": "protagonist|ally|antagonist|mentor|neutral",
      "traits": ["2-4 personality traits shown through ACTIONS"],
      "appearance": "string - physical description (colors, features, clothing)"
    }
  ],
  "beats": [
    {
      "beat_type": "setup|inciting_incident|rising_action|crisis|climax|resolution",
      "summary": "string - what happens in this story segment",
      "timestamp_range": [start_seconds, end_seconds]
    }
  ],
  "moments": [
    {
      "moment_id": "string - unique identifier",
      "beat_type": "string - which story beat this belongs to",
      "timestamp_range": [start_seconds, end_seconds],
      "visual_description": "string - VIVID, DETAILED description (see rules below)",
      "key_dialogue": ["exact spoken lines from this moment"],
      "emotional_beat": "string - single emotion (excitement, fear, triumph, etc.)",
      "screenshot_candidates_s": [2-4 precise timestamps for clear frames]
    }
  ]
}

VISUAL DESCRIPTION RULES (CRITICAL):
Your visual_description must include ALL of these elements when present:

1. CHARACTER ACTIONS: What are they physically doing? (gripping, running, jumping, pointing)
2. FACIAL EXPRESSIONS: What emotions show on their faces? (eyes wide, grinning, frowning)
3. BODY LANGUAGE: Posture, gestures, movement (leaning forward, arms raised, crouching)
4. ENVIRONMENT: Where are they? What's around them?
5. MOTION & SPEED: How fast? What direction?
6. COLORS & VISUALS: Important colors, effects, lighting
7. SOUNDS IMPLIED: What would we hear?
8. CAMERA ANGLE: Close-up, wide shot, action shot (helps writer set scene)

{franchise_visual_examples}

OUTPUT: Return ONLY valid JSON. No markdown, no explanations."""


VIDEO_ANALYSIS_USER = """Analyze this video episode for a children's picture book (ages 5-8, ~22 pages).

VIDEO: {video_filename}
TOTAL DURATION: {duration_s:.1f} seconds (IMPORTANT: analyze the ENTIRE video from 0 to {duration_s:.0f} seconds)

SUBTITLES FOR REFERENCE:
{subtitles_text}

EXTRACTION TASKS:

1. TITLE CANDIDATES (3-5)
   - Engaging, action-oriented titles suitable for children
   - Based on the main story arc you observe

2. CHARACTERS (CRITICAL - include physical appearance)
   - Every speaking/major character you SEE
   - APPEARANCE: hair color/style, skin tone, clothing colors, gender presentation (he/she/they)
   - Example: "{franchise_character_example}"
   - Personality shown through their ACTIONS (not told)

3. STORY BEATS (8-12)
   - Break the ENTIRE episode into narrative segments
   - Beats MUST span from 0 seconds to {duration_s:.0f} seconds
   - Include timestamp ranges for each beat

4. KEY MOMENTS (25-35 moments - CRITICAL)
   - You MUST provide 25-35 moments spread across the ENTIRE video
   - Timestamps MUST span from early (0-100s) to middle (100-400s) to late (400-{duration_s:.0f}s)
   - DO NOT cluster all moments in the first 2 minutes - spread them throughout!
   - MUST include vivid visual descriptions (50-150 words each)
   - Include key dialogue from subtitles
   - Provide 2-4 screenshot timestamp candidates per moment

TIMESTAMP COVERAGE CHECK:
- Early moments (0-{early_third:.0f}s): at least 8 moments
- Middle moments ({early_third:.0f}-{late_third:.0f}s): at least 10 moments  
- Late moments ({late_third:.0f}-{duration_s:.0f}s): at least 8 moments

REMEMBER: The story writer will NEVER see this video. Your descriptions are their eyes. Be vivid, specific, and detailed."""


# =============================================================================
# STORY WRITING PROMPT (V6 - Example-Driven Style)
# =============================================================================

STORY_WRITING_SYSTEM = """You are writing a children's picture book for ages 5-8.

═══════════════════════════════════════════════════════════════════════════════
STYLE GUIDE: Study these example paragraphs
═══════════════════════════════════════════════════════════════════════════════

Below are example paragraphs showing the TARGET STYLE for this franchise. 
Notice the paragraph length, vocabulary, and how dialogue is integrated.

{franchise_style_examples}

═══════════════════════════════════════════════════════════════════════════════
CHARACTER PRONOUNS (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

You will receive a CHARACTER REFERENCE BLOCK with each character's pronouns.
You MUST use these pronouns EXACTLY. This is NON-NEGOTIABLE.

{franchise_pronoun_example}

═══════════════════════════════════════════════════════════════════════════════
KEY STYLE RULES
═══════════════════════════════════════════════════════════════════════════════

1. PARAGRAPH LENGTH: 30-50 words each. Never single sentences alone.
2. DIALOGUE INTEGRATION: Always wrap dialogue with action/context.
   ❌ "Let's go!" She ran.
   ✅ "Let's go!" she called, and they all ran together.
3. VOCABULARY: Simple words
4. CATCHPHRASES: Wrap signature phrases in [bold] tags at emotional peaks
5. PACING: Each paragraph = one "beat" (one moment or mini-scene)
6. TOTAL LENGTH: ~1,100-1,200 words, ~28-32 paragraphs

{franchise_style_rules}

═══════════════════════════════════════════════════════════════════════════════
STORY STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

OPENING (~15%): Hook the reader, establish world and main character
SETUP (~25%): Introduce mentor + ALL other characters, explain the situation
RISING ACTION (~25%): Problem emerges, characters work together
CLIMAX (~20%): The breakthrough - give this SPACE, multiple paragraphs!
RESOLUTION (~15%): Lesson learned, celebration, hint at next adventure

OUTPUT: Markdown with double newlines between paragraphs."""


STORY_WRITING_USER = """Write a children's picture book story (~1,100 words) based on this episode.
Match the style from the examples: paragraph length, dialogue integration, simple vocabulary.

═══════════════════════════════════════════════════════════════════════════════
FRANCHISE CONTEXT
═══════════════════════════════════════════════════════════════════════════════
{franchise_context}

═══════════════════════════════════════════════════════════════════════════════
CHARACTERS (use these exact names and pronouns)
═══════════════════════════════════════════════════════════════════════════════
{character_block}

═══════════════════════════════════════════════════════════════════════════════
CATCHPHRASES (include with [bold] formatting)
═══════════════════════════════════════════════════════════════════════════════
{catchphrases_block}

═══════════════════════════════════════════════════════════════════════════════
EPISODE EVENTS (follow this sequence faithfully)
═══════════════════════════════════════════════════════════════════════════════
{analysis_json}

═══════════════════════════════════════════════════════════════════════════════
EPISODE DIALOGUE (use actual lines where they fit naturally)
═══════════════════════════════════════════════════════════════════════════════
{subtitles_text}

═══════════════════════════════════════════════════════════════════════════════

Write the story now. Match the example's style exactly.

REMINDER: ~1,100-1,200 words total (~28-32 paragraphs)."""


# =============================================================================
# PAGINATION PROMPT (MiMo)
# =============================================================================

PAGINATION_SYSTEM = """You are a picture book layout engine that understands story context.

YOUR TASK:
1. Read the story and identify which ANALYSIS MOMENT each paragraph describes
2. Group paragraphs into CONTEXT CLUMPS (coherent scenes/beats)
3. Assign each page a moment_id linking it to the analysis

OUTPUT SCHEMA:
{{
  "schema_version": "1.0",
  "title": "string",
  "pages": [
    {{
      "page_index": 1,
      "paragraphs": ["First paragraph text...", "Second paragraph text..."],
      "moment_id": "string - which analysis moment this page depicts",
      "layout_hint": "auto"
    }}
  ]
}}

NOTE: Do NOT include image_timestamp_candidates_s - timestamps will be assigned 
automatically by matching dialogue to subtitles (more accurate than moment-based).

CONTEXT CLUMPING RULES:
1. Each page = ONE coherent scene or emotional beat
2. Keep cause-and-effect together (action + reaction on same page)
3. Scene changes = new page
4. Dialogue exchanges can span 1-2 pages max

MOMENT MATCHING (CHRONOLOGICAL ORDER REQUIRED):
- The story flows FORWARD in time - moment_ids must also progress forward
- Page 1 might use moment_005, Page 2 might use moment_008, Page 3 might use moment_012
- NEVER go backwards: if Page 5 uses moment_020, Page 6 CANNOT use moment_015
- Read story paragraph → find matching moment → verify it's AFTER the previous page's moment
- If exact match would violate chronology, pick the NEXT closest moment instead
- Story about "tire bouncing" → find moment about "runaway tire" → verify timestamp is later than previous

TEXT FORMATTING (PRESERVE EXACTLY):
- KEEP all [bold]text[/bold] markers exactly as written
- KEEP all [italic]text[/italic] markers exactly as written
- These markers are PART OF THE TEXT - do NOT remove them
- Example: "A racer always says [bold]\"Challenge accepted!\"[/bold]" stays EXACTLY as written

PAGE REQUIREMENTS:
1. PAGE COUNT: Target {target_pages} pages (range: {min_pages}-{max_pages})
2. WORDS PER PAGE: {words_per_page} words average
3. PARAGRAPHS PER PAGE: 1-3 paragraphs per page (use multiple when they form one scene)
4. LAYOUT: Set layout_hint to "auto" (will be finalized by post-processor)

OUTPUT: Valid JSON only. No markdown."""


PAGINATION_USER = """Create a paginated picture book by matching story content to analysis moments.

STORY TEXT:
{story_text}

ANALYSIS MOMENTS (for moment_id matching):
{analysis_json}

CRITICAL REQUIREMENTS:
1. TARGET {target_pages} PAGES (range: {min_pages}-{max_pages})
2. PRESERVE ALL [bold] and [italic] FORMATTING exactly as written in the story
3. Each page should have 1-3 paragraphs ({words_per_page} words total per page)
4. DO NOT strip or modify any text - copy paragraphs EXACTLY

PROCESS:
1. For each story paragraph, identify which analysis moment it describes
2. Group related paragraphs into pages ({words_per_page} words each)
3. Assign each page the moment_id of its matching moment
4. VERIFY: Each page's moment_id must have a LATER timestamp than the previous page
5. If a moment would go backwards in time, skip to the next available moment

{franchise_pagination_examples}

NOTE: Do NOT include image_timestamp_candidates_s in your output.
Timestamps will be assigned automatically from dialogue matching.

Create the pagination now with proper moment_id matching."""


# =============================================================================
# SCREENSHOT SELECTION PROMPTS (MiMo + MLX-VLM)
# =============================================================================

# Stage 1: Visual target generation (MiMo; JSON output)
VISUAL_TARGETS_SYSTEM = """You are a picture book art director selecting the best screenshot for each page.

You will be given (a) the paginated pages of a children's book and (b) the video analysis moments.

Your job: for EACH page, describe what the ideal screenshot should show.

GUIDELINES:
- Describe the scene simply: who/what should be visible and what they're doing
- Name specific characters by name when possible
- Focus on the KEY MOMENT that matches the text
- Be flexible - describe what you WANT, not strict requirements

Return STRICT JSON only (no markdown)."""

VISUAL_TARGETS_USER = """Generate visual targets for these pages.

PAGES_JSON:
{pages_json}

ANALYSIS_JSON (moments, beats, characters):
{analysis_json}

OUTPUT SCHEMA (Strict JSON):
{{
  "targets": [
    {{
      "page_index": 1,
      "visual_target": "string - simple description of ideal screenshot",
      "key_dialogue": ["string", "..."],
      "key_actions": ["string", "..."]
    }}
  ]
}}

RULES:
1. Create exactly one entry per page_index in the provided PAGES_JSON.
2. visual_target should be a simple, clear description (1-2 sentences)
3. Focus on WHAT should be visible, not camera angles

{franchise_visual_guidance}

{franchise_visual_target_examples}
"""


# Stage 3: Frame scoring (MLX-VLM; image + text prompt)
#
# Opinionated contract: the model MUST return only a single number 1-10.
VLM_FRAME_SCORE_PROMPT = """Rate how well this image matches the visual target for a children's picture book.

Visual target:
"{visual_target}"

Score from 1-10:
- 9-10: Perfect match - shows exactly what's described
- 7-8: Good match - shows most of what's described  
- 5-6: Partial match - shows some elements but missing others
- 3-4: Weak match - related scene but different moment
- 1-2: Poor match - wrong scene entirely

Return ONLY a single number from 1 to 10. No words, no explanation."""

