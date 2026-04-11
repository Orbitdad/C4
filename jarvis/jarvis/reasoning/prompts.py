from __future__ import annotations

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
- For React Vite apps: include package.json, index.html, vite config, src/main, src/App, routing/login page/components, minimal styles.
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
{
  "strategy": "patch" | "rewrite" | "noop",
  "target_file": "<relative path or empty>",
  "reason": "<short string>",
  "notes": "<short string>"
}
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

