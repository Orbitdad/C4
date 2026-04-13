from __future__ import annotations

# ── C4 Web Design System ──────────────────────────────────────────────────────
# Injected verbatim into every web file generation prompt so local LLMs have a
# concrete visual reference instead of guessing at "glassmorphism" abstractions.
C4_WEB_DESIGN_SYSTEM = """
/* ═══════════════════════════════════════════════════════
   C4 DESIGN SYSTEM — embed this block in every web page
   ═══════════════════════════════════════════════════════ */

/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
  /* Brand palette */
  --c-accent:       hsl(200, 100%, 60%);   /* electric cyan  */
  --c-accent2:      hsl(260, 90%,  65%);   /* violet         */
  --c-glow:         hsl(200, 100%, 70%);   /* hover glow     */
  --c-bg:           hsl(225, 25%,  7%);    /* near-black     */
  --c-surface:      hsl(225, 20%, 12%);    /* card base      */
  --c-surface2:     hsl(225, 18%, 17%);    /* elevated card  */
  --c-border:       hsl(220, 20%, 22%);    /* subtle border  */
  --c-text:         hsl(220, 15%, 92%);    /* primary text   */
  --c-muted:        hsl(220, 10%, 55%);    /* secondary text */

  /* Typography */
  --font-display:   'Outfit', sans-serif;
  --font-body:      'Inter', sans-serif;

  /* Radii */
  --r-sm: 8px;  --r-md: 14px;  --r-lg: 24px;  --r-xl: 36px;

  /* Shadows */
  --shadow-glow: 0 0 40px hsla(200,100%,60%,0.25);
  --shadow-card: 0 8px 32px hsla(220,25%,5%,0.6);

  /* Transitions */
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
  --dur: 280ms;
}

/* Keyframes */
@keyframes fadeUp   { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:none; } }
@keyframes gradShift{ 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 hsla(200,100%,60%,0.4)} 50%{box-shadow:0 0 0 12px transparent} }
@keyframes float    { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }

/* Glassmorphism utility */
.glass {
  background: hsla(220, 20%, 15%, 0.55);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  border: 1px solid hsla(220, 30%, 50%, 0.18);
  border-radius: var(--r-lg);
}

/* Gradient text utility */
.grad-text {
  background: linear-gradient(135deg, var(--c-accent), var(--c-accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Animated gradient background */
.grad-bg {
  background: linear-gradient(135deg, var(--c-bg), hsl(240,25%,10%), hsl(200,25%,9%));
  background-size: 300% 300%;
  animation: gradShift 12s ease infinite;
}
"""


PROJECT_PLAN_PROMPT = """You are C4 Planner.

Generate a STRICT JSON object that follows this schema EXACTLY (no extra keys, no markdown, no commentary):

{{
  "schema_version": 1,
  "intent": "create_project" | "modify_project",
  "project": {{
    "name": "<string>",
    "type": "web" | "android" | "python",
    "stack": "<string>",
    "package_manager": "npm" | "pnpm" | "yarn" | "gradle" | "pip" | "none",
    "language": "typescript" | "javascript" | "kotlin" | "java" | "python"
  }},
  "workspace": {{
    "root": "<string path or '.'>",
    "assume_existing": true | false
  }},
  "files": [
    {{
      "path": "<relative path from workspace.root>",
      "mode": "create" | "update" | "patch",
      "purpose": "<short string>",
      "depends_on": ["<relative paths>"]
    }}
  ],
  "commands": [
    {{
      "cmd": "<shell command string>",
      "requires_confirmation": true | false,
      "priority": "critical" | "high" | "normal" | "low",
      "retries": 0,
      "delay_seconds": 0,
      "cancel_token": "<string or empty>",
      "purpose": "<short string>"
    }}
  ],
  "acceptance_criteria": [
    "<string>"
  ]
}}

Rules:
- Output ONLY JSON that parses.
- Use relative paths. Do not use absolute paths.
- Prefer "update" for existing files, "create" for missing.
- Use "patch" only if a small localized change is sufficient; otherwise use "update".
- STACK PREFERENCE: 
    - For simple web requests, prefer Vanilla HTML, CSS, and Javascript (beginner friendly).
    - Use React/Vite ONLY if explicitly asked or for complex stateful applications.
- DESIGN STANDARDS:
    - All web projects must use a premium, modern design system.
    - Include vibrant colors, glassmorphism, smooth gradients, and micro-animations.
    - Use Google Fonts (e.g., Inter, Roboto, Outfit) instead of browser defaults.
- Keep file list complete enough that the project can run after commands.

Request:
{request}

Workspace context:
{workspace_context}
"""


FILE_GENERATION_PROMPT = """You are C4 Coder.

Task: Write the FULL content of the file shown in the JSON below.

Output rules (STRICT):
- Output ONLY the file content. No markdown fences. No commentary.
- The content must be complete and syntactically valid.
- Preserve project conventions from the provided context.
- CODE QUALITY (CRITICAL): You MUST output the absolute BEST, most optimal, advanced but structurally readable code. Implement comprehensive error handling, modularity, and follow modern best practices perfectly.
- DESIGN AESTHETICS (CRITICAL — for HTML/CSS/JS files):
    - You MUST embed the C4 Design System variables and utilities below into every HTML file via a <style> block.
    - Use rich aesthetics: the CSS variables (--c-accent, --c-surface, etc.), gradient text, glassmorphism cards.
    - Apply fadeUp animations to section entries (animation: fadeUp 0.6s var(--ease) both).
    - Every page must use Outfit for headings and Inter for body text (loaded from Google Fonts already in design system).
    - The design must feel premium, futuristic, and production-ready.

C4 Design System (embed in every HTML/CSS file):
{design_system}

Target file JSON:
{file_spec_json}

Project plan JSON:
{project_plan_json}

Workspace context bundle:
{context_bundle}

If you need to reference other files, do so by importing/using them appropriately, but still output ONLY the target file content.
"""


FILE_MODIFY_FULL_REWRITE_PROMPT = """You are C4 Coder.

Task: Modify the given EXISTING file to implement the request, keeping style consistent.

Output rules (STRICT):
- Output ONLY the full updated file content. No markdown fences. No commentary.
- Keep unrelated code unchanged as much as possible.
- Ensure imports compile and formatting is consistent.
- CODE QUALITY (CRITICAL): Ensure the modified code meets the highest standards. Implement robust error handling, edge-case protection, and best practices seamlessly into the existing structure.
- DESIGN AESTHETICS (CRITICAL — for HTML/CSS/JS files): The C4 Design System must remain embedded. Do NOT remove CSS variables or animation keyframes.

C4 Design System reference (keep embedded — do NOT remove from HTML/CSS files):
{design_system}

Request:
{request}

Target file path:
{path}

Project plan JSON:
{project_plan_json}

Workspace context bundle:
{context_bundle}

EXISTING FILE CONTENT (authoritative):
{existing_file}
"""


FILE_MODIFY_UNIFIED_DIFF_PROMPT = """You are C4 Coder.

Task: Produce a UNIFIED DIFF patch for the given EXISTING file to implement the request.

Output rules (STRICT):
- Output ONLY a unified diff. No markdown fences. No commentary.
- Patch must apply cleanly with exact context.
- Use minimal changes.
- The diff must be for this exact path and only this file.

Request:
{request}

Target file path:
{path}

Project plan JSON:
{project_plan_json}

Workspace context bundle:
{context_bundle}

EXISTING FILE CONTENT (authoritative):
{existing_file}
"""


DEBUG_FIX_ERRORS_PROMPT = """You are C4 Debugger.

Fix build/runtime errors using the supplied logs and code context.

Output rules (STRICT):
- Return ONLY JSON with this schema:
{{
  "strategy": "patch" | "rewrite" | "noop",
  "target_file": "<relative path or empty>",
  "reason": "<short string>",
  "notes": "<short string>"
}}
- If errors cannot be fixed from context, return strategy "noop".
- Prefer "patch" for localized fixes.

User request:
{request}

Project plan JSON:
{project_plan_json}

Structured errors JSON:
{errors_json}

Context bundle JSON:
{context_bundle}
"""


DEBUG_UNIFIED_DIFF_PROMPT = """You are C4 Debugger.

Generate a unified diff patch for the target file that fixes the listed errors.

Output rules (STRICT):
- Output ONLY unified diff text (no markdown, no commentary).
- Include correct @@ hunks with enough context.
- Modify only the target file.
- Keep changes minimal and compile-safe.

Target file:
{target_file}

User request:
{request}

Structured errors JSON:
{errors_json}

Project plan JSON:
{project_plan_json}

Context bundle JSON:
{context_bundle}

Current target file content:
{existing_file}
"""

WEBSITE_QUALITY_CHECK_PROMPT = """You are C4 QA.

Review the HTML file content below and answer ONLY with this JSON (no commentary):
{{
  "pass": true | false,
  "score": 0-100,
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "target_file": "<relative path>",
  "improvements": "<one paragraph of concrete CSS/HTML changes to apply>"
}}

Pass criteria (score >= 75 required to pass). Fail if ANY of these are missing:
- Google Fonts or custom @import font loaded
- CSS custom properties (:root variables) defined
- At least one CSS animation or transition
- Responsive layout using flexbox or CSS grid
- Dark background with light text (sufficient contrast)
- An <h1> heading and at least one <section> element

HTML content to review:
{html_content}

Relative path of this file:
{rel_path}
"""
