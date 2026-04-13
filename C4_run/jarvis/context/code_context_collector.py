from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    text: str
    sha1: str


@dataclass(frozen=True)
class ContextItem:
    path: str
    reason: str
    score: float


class CodeContextCollector:
    """
    Collects relevant repo context (file list + ranked snippets) for code generation/modification.

    Designed to work with:
    - local filesystem (discovery + reading)
    - SemanticDB (optional) for chunk retrieval
    """

    DEFAULT_IGNORE_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".expo",
        ".gradle",
        ".dart_tool",
        "target",
        ".venv",
        "venv",
    }

    DEFAULT_TEXT_EXTS = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".md",
        ".css",
        ".scss",
        ".html",
        ".yml",
        ".yaml",
        ".py",
        ".java",
        ".kt",
        ".gradle",
        ".properties",
        ".xml",
        ".toml",
        ".txt",
    }

    def __init__(
        self,
        workspace_root: Path,
        semantic_db: Optional[Any] = None,
        ignore_dirs: Optional[Iterable[str]] = None,
        text_exts: Optional[Iterable[str]] = None,
        max_file_bytes: int = 512_000,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.semantic_db = semantic_db
        self.ignore_dirs = set(ignore_dirs or self.DEFAULT_IGNORE_DIRS)
        self.text_exts = set(text_exts or self.DEFAULT_TEXT_EXTS)
        self.max_file_bytes = max_file_bytes

    # ---------- Discovery ----------

    def discover_files(self) -> List[Path]:
        files: List[Path] = []
        root = self.workspace_root
        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored dirs
            dirnames[:] = [d for d in dirnames if d not in self.ignore_dirs]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in self.text_exts:
                    continue
                try:
                    if p.stat().st_size > self.max_file_bytes:
                        continue
                except OSError:
                    continue
                files.append(p)
        return files

    def detect_project_type(self, files: Optional[List[Path]] = None) -> str:
        files = files or self.discover_files()
        names = {p.name.lower() for p in files}
        if "package.json" in names:
            return "node"
        if "build.gradle" in names or "settings.gradle" in names or "gradlew" in names:
            return "android"
        if "pyproject.toml" in names or "requirements.txt" in names:
            return "python"
        return "unknown"

    # ---------- Chunking / indexing ----------

    def chunk_file(self, path: Path, chunk_lines: int = 120, overlap: int = 20) -> List[CodeChunk]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        lines = text.splitlines()
        if not lines:
            return []

        rel = str(path.resolve().relative_to(self.workspace_root)).replace("\\", "/")

        chunks: List[CodeChunk] = []
        step = max(1, chunk_lines - overlap)
        for start in range(0, len(lines), step):
            end = min(len(lines), start + chunk_lines)
            chunk_text = "\n".join(lines[start:end])
            sha = hashlib.sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()
            chunks.append(
                CodeChunk(
                    path=rel,
                    start_line=start + 1,
                    end_line=end,
                    text=chunk_text,
                    sha1=sha,
                )
            )
            if end >= len(lines):
                break
        return chunks

    def index_workspace(self, max_files: int = 200, chunk_lines: int = 120, overlap: int = 20) -> Dict[str, Any]:
        """
        Add code chunks to SemanticDB. Idempotence is best-effort via chunk sha.
        Returns summary stats.
        """
        if not self.semantic_db:
            return {"indexed": 0, "reason": "semantic_db_not_configured"}

        files = self.discover_files()
        # prioritize smaller, likely-entry files
        files.sort(key=lambda p: (0 if p.name in ("package.json", "README.md") else 1, p.stat().st_size if p.exists() else 0))
        files = files[:max_files]

        indexed = 0
        for p in files:
            for ch in self.chunk_file(p, chunk_lines=chunk_lines, overlap=overlap):
                meta = {
                    "type": "code_chunk",
                    "path": ch.path,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "sha1": ch.sha1,
                }
                self.semantic_db.add_memory(ch.text, metadata=meta)
                indexed += 1
        return {"indexed": indexed, "files": len(files)}

    # ---------- Ranking / retrieval ----------

    def rank_files_for_request(self, request: str, files: Optional[List[Path]] = None, top_k: int = 30) -> List[ContextItem]:
        """
        Heuristic ranking for relevant files before any semantic search.
        """
        req = (request or "").lower()
        files = files or self.discover_files()

        scored: List[ContextItem] = []
        for p in files:
            rel = str(p.resolve().relative_to(self.workspace_root)).replace("\\", "/")
            name = p.name.lower()
            score = 0.0

            # common entrypoints
            if name in {"package.json", "vite.config.ts", "vite.config.js", "index.html", "src/main.tsx", "src/main.jsx"}:
                score += 3.0

            # request keywords
            if "login" in req and any(k in rel.lower() for k in ["login", "auth", "signin", "signup"]):
                score += 4.0
            if "vite" in req and "vite" in name:
                score += 2.0
            if "react" in req and any(k in rel.lower() for k in ["react", "tsx", "jsx"]):
                score += 1.5

            # bias toward src/
            if rel.startswith("src/"):
                score += 1.0

            # deprioritize tests
            if any(seg in rel.lower() for seg in ["/test", "/tests", "__tests__"]):
                score -= 0.5

            if score > 0:
                scored.append(ContextItem(path=rel, reason="heuristic_match", score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def retrieve_snippets(self, query: str, top_k: int = 8) -> List[CodeChunk]:
        """
        Retrieve best matching chunks using SemanticDB.
        Expects SemanticDB documents to include metadata.type == 'code_chunk'.
        """
        if not self.semantic_db or not query:
            return []

        matches = self.semantic_db.search(query, top_k=max(12, top_k * 2))
        chunks: List[CodeChunk] = []
        for sim, doc in matches:
            meta = (doc or {}).get("metadata") or {}
            if meta.get("type") != "code_chunk":
                continue
            text = doc.get("text") or ""
            if not text:
                continue
            chunks.append(
                CodeChunk(
                    path=str(meta.get("path") or ""),
                    start_line=int(meta.get("start_line") or 1),
                    end_line=int(meta.get("end_line") or 1),
                    text=text,
                    sha1=str(meta.get("sha1") or ""),
                )
            )
            if len(chunks) >= top_k:
                break
        return chunks

    def load_file_text(self, rel_path: str) -> str:
        p = (self.workspace_root / rel_path).resolve()
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def build_context_bundle(
        self,
        request: str,
        max_ranked_files: int = 12,
        max_snippets: int = 8,
        include_full_files: int = 3,
    ) -> Dict[str, Any]:
        """
        Returns a prompt-ready context bundle:
        - discovered project type
        - ranked file paths
        - top semantic snippets (if available)
        - full contents for a few key files
        """
        files = self.discover_files()
        project_type = self.detect_project_type(files)
        ranked = self.rank_files_for_request(request, files=files, top_k=max_ranked_files)
        snippets = self.retrieve_snippets(request, top_k=max_snippets)

        full_texts: List[Tuple[str, str]] = []
        for item in ranked[: max(0, include_full_files)]:
            txt = self.load_file_text(item.path)
            if txt:
                full_texts.append((item.path, txt))

        return {
            "workspace_root": str(self.workspace_root),
            "project_type": project_type,
            "ranked_files": [i.__dict__ for i in ranked],
            "snippets": [c.__dict__ for c in snippets],
            "full_files": [{"path": p, "text": t} for p, t in full_texts],
        }

