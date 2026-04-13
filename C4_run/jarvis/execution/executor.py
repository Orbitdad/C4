"""
Safe execution of actions: open apps, file ops, web, etc.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    pyautogui = None

from jarvis.logging_utils import log_action


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Executor:
    """
    Executes action steps with safety checks, dry-run support, and logging.
    """

    def __init__(
        self,
        dry_run: bool = False,
        confirm_deletes: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self.confirm_deletes = confirm_deletes

    def execute_step(self, step: Any) -> Dict[str, Any]:
        """
        Execute a single step. Step must have .type and .params.
        Returns {"success": bool, "message": str, ...}.
        """
        step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
        params = getattr(step, "params", None) or (step if isinstance(step, dict) else {})
        if isinstance(step, dict):
            params = {k: v for k, v in step.items() if k != "type"}
        else:
            params = getattr(step, "params", {})

        if self.dry_run:
            log_action("dry_run", {"step": step_type, "params": params})
            return {"success": True, "message": f"[Dry run] Would execute {step_type}"}

        if step_type == "open_app":
            return self._open_app(params)
        if step_type == "open_url":
            return self._open_url(params)
        if step_type == "create_file":
            return self._create_file(params)
        if step_type == "read_file":
            return self._read_file(params)
        if step_type == "update_file":
            return self._update_file(params)
        if step_type == "patch_file":
            return self._patch_file(params)
        if step_type == "delete_file":
            return self._delete_file(params)
        if step_type == "tell_time":
            return self._tell_time()
        if step_type == "tell_date":
            return self._tell_date()
        if step_type == "web_search":
            return self._web_search(params)
        if step_type == "play_media":
            return self._play_media(params)
        if step_type == "run_command":
            return self._run_command(params)
        if step_type == "manage_window":
            return self._manage_window(params)
        if step_type == "run_python":
            return self._run_python(params)
            
        # Hardware / UI Control
        if step_type == "mouse_move":
            return self._mouse_move(params)
        if step_type == "mouse_click":
            return self._mouse_click(params)
        if step_type == "keyboard_type":
            return self._keyboard_type(params)
        if step_type == "keyboard_hotkey":
            return self._keyboard_hotkey(params)

        return {"success": False, "message": f"Unknown step type: {step_type}"}

    def _open_app(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import string
        import shutil
        app = params.get("app") or params.get("query", "")
        if not app:
            return {"success": False, "message": "No application specified."}
            
        app_clean = app.translate(str.maketrans("", "", string.punctuation)).strip().lower()
        
        # Chrome Profile Integration
        is_chrome = any(name in app_clean for name in ["chrome", "google chrome"])
        profile_dir = None
        if is_chrome:
            profiles = self._get_chrome_profiles()
            # Try to find a profile name in the query
            for p_name, p_dir in profiles.items():
                if p_name in app_clean:
                    profile_dir = p_dir
                    break

        # Map common names to executables
        mapping = {
            "code": "code",
            "vscode": "code",
            "vs code": "code",
            "google chrome": "chrome",
            "chrome": "chrome" if platform.system() == "Windows" else "google-chrome",
            "firefox": "firefox",
            "notepad": "notepad" if platform.system() == "Windows" else "xdg-open",
            "calculator": "calc",
            "calc": "calc",
            "command prompt": "cmd",
            "terminal": "wt" if platform.system() == "Windows" else "x-terminal-emulator",
            "explorer": "explorer",
            "file explorer": "explorer",
        }
        exe = mapping.get(app_clean, app_clean)
        
        # If it's chrome and we still have the full app name as 'chrome', 
        # but a profile was found, ensure we use 'chrome'
        if is_chrome:
            exe = "chrome" if platform.system() == "Windows" else "google-chrome"

        try:
            if platform.system() == "Windows":
                # Special handling for Chrome profiles
                if is_chrome and profile_dir:
                    cmd = f'start chrome --profile-directory="{profile_dir}"'
                    subprocess.Popen(cmd, shell=True)
                # Standard app opening
                elif exe.endswith(".exe") or shutil.which(exe):
                    os.startfile(exe) if "." in exe else subprocess.Popen([exe], shell=True)
                else:
                    # Windows 'start' searches registry App Paths, useful for unmapped installed software
                    subprocess.Popen(f"start {exe}", shell=True)
            else:
                # Linux/Mac
                cmd_args = [exe]
                if is_chrome and profile_dir:
                    cmd_args.append(f'--profile-directory={profile_dir}')
                subprocess.Popen(cmd_args, start_new_session=True)
                
            log_action("open_app", {"app": app, "profile": profile_dir})
            return {"success": True, "message": f"Opened {app}." if not profile_dir else f"Opening {app} using the {profile_dir} profile."}
        except Exception as e:
            return {"success": False, "message": f"Failed to open {app}. Error: {str(e)}"}

    def _get_chrome_profiles(self) -> Dict[str, str]:
        """Maps lowercase display names to internal directory names (Windows/Linux/Mac)."""
        import os
        import json
        profiles_map = {}
        try:
            if platform.system() == "Windows":
                path = os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data\Local State')
            elif platform.system() == "Darwin":
                path = os.path.expanduser('~/Library/Application Support/Google/Chrome/Local State')
            else:
                path = os.path.expanduser('~/.config/google-chrome/Local State')

            if not os.path.exists(path):
                return {}

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info_cache = data.get('profile', {}).get('info_cache', {})
            for dir_name, info in info_cache.items():
                display_name = info.get('name', '').lower()
                if display_name:
                    profiles_map[display_name] = dir_name
        except Exception as e:
            logger.debug(f"Failed to parse Chrome profiles: {e}")
        return profiles_map

    def _open_url(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url", "")
        if not url:
            return {"success": False, "message": "No URL specified."}
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            if platform.system() == "Windows":
                os.startfile(url)
            else:
                subprocess.Popen(["xdg-open", url], start_new_session=True)
            log_action("open_url", {"url": url})
            return {"success": True, "message": f"Opened {url}."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _create_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path") or params.get("query", "new_file.txt")
        content = params.get("content", "")
        path = path.strip()
        if not path:
            return {"success": False, "message": "No path specified."}
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log_action("create_file", {"path": str(p), "content_len": len(content)})
            return {"success": True, "message": f"Created {p} with {len(content)} bytes."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _update_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely overwrite an existing file (or create it if missing) with new content.
        Optional safety: require 'expected_sha256' to match the current file content.
        """
        path = (params.get("path") or params.get("query") or "").strip()
        content = params.get("content", "")
        expected_sha256 = (params.get("expected_sha256") or "").strip()
        if not path:
            return {"success": False, "message": "No path specified."}

        p = Path(path)
        try:
            # Check optimistic concurrency if requested
            if expected_sha256 and p.is_file():
                import hashlib
                current = p.read_bytes()
                current_sha = hashlib.sha256(current).hexdigest()
                if current_sha != expected_sha256:
                    return {"success": False, "message": f"Refusing to overwrite {p}: file changed (sha mismatch)."}

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log_action("update_file", {"path": str(p), "content_len": len(content)})
            return {"success": True, "message": f"Updated {p} ({len(content)} bytes)."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _patch_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a unified diff patch to a file (single-file patches recommended).
        This is optional and best-effort: it requires exact context match.
        """
        path = (params.get("path") or "").strip()
        diff_text = params.get("diff") or params.get("patch") or ""
        if not path:
            return {"success": False, "message": "No path specified."}
        if not diff_text.strip():
            return {"success": False, "message": "No diff provided."}

        p = Path(path)
        if not p.is_file():
            return {"success": False, "message": f"File not found: {p}."}

        try:
            original_text = p.read_text(encoding="utf-8", errors="replace")
            original = original_text.splitlines(keepends=False)
            patched = _apply_unified_diff_to_lines(original, diff_text)
            if patched is None:
                return {"success": False, "message": "Patch could not be applied (context mismatch)."}
            # Preserve trailing newline if the original had one.
            trailing_nl = original_text.endswith("\n")
            out_text = "\n".join(patched) + ("\n" if trailing_nl else "")
            p.write_text(out_text, encoding="utf-8")
            log_action("patch_file", {"path": str(p), "diff_len": len(diff_text)})
            return {"success": True, "message": f"Patched {p}."}
        except Exception as e:
            return {"success": False, "message": f"Patch error: {e}"}

    def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path") or params.get("query", "")
        if not path:
            return {"success": False, "message": "No path specified."}
        p = Path(path)
        if not p.is_file():
            return {"success": False, "message": f"File not found: {p}."}
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            preview = text[:500] + "..." if len(text) > 500 else text
            log_action("read_file", {"path": str(p)})
            return {"success": True, "message": f"Contents: {preview}", "full_text": text}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path") or params.get("query", "")
        if not path:
            return {"success": False, "message": "No path specified."}
        p = Path(path)
        if not p.is_file():
            return {"success": False, "message": f"File not found: {p}."}
        if self.confirm_deletes:
            # Caller should have confirmed; we proceed
            pass
        try:
            p.unlink()
            log_action("delete_file", {"path": str(p)})
            return {"success": True, "message": f"Deleted {p}."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _tell_time(self) -> Dict[str, Any]:
        from datetime import datetime
        now = datetime.now()
        msg = now.strftime("%I:%M %p")
        return {"success": True, "message": f"The time is {msg}."}

    def _tell_date(self) -> Dict[str, Any]:
        from datetime import datetime
        now = datetime.now()
        msg = now.strftime("%A, %B %d, %Y")
        return {"success": True, "message": f"Today is {msg}."}

    def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        if not query:
            return {"success": False, "message": "No search query."}
        # Open search in browser
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        try:
            if platform.system() == "Windows":
                os.startfile(url)
            else:
                subprocess.Popen(["xdg-open", url], start_new_session=True)
            log_action("web_search", {"query": query})
            return {"success": True, "message": f"Searching for {query}."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _play_media(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        url = params.get("url", "")
        if url:
            return self._open_url({"url": url})
        if query:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            return self._open_url({"url": search_url})
        return {"success": False, "message": "No media query."}

    def _run_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cmd = params.get("command", params.get("cmd", ""))
        cwd = params.get("cwd")
        if not cmd:
            return {"success": False, "message": "No command specified."}
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd if cwd else None,
            )
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            out = (stdout or stderr or "").strip()[:1200]
            log_action("run_command", {"command": cmd[:100]})
            msg = out if out else ("Command completed successfully." if result.returncode == 0 else "Command failed with no output.")
            return {
                "success": result.returncode == 0,
                "message": msg,
                "returncode": result.returncode,
                "stdout": stdout[:20000],
                "stderr": stderr[:20000],
                "command": cmd,
                "cwd": cwd or "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "message": f"Command '{cmd}' timed out after 30 seconds."}
        except Exception as e:
            return {"success": False, "message": f"Execution error: {str(e)}"}

    def _manage_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses ctypes to natively manipulate windows on Windows OS.
        """
        if platform.system() != "Windows":
            return {"success": False, "message": "Window management currently only supported on Windows."}
            
        target = params.get("target", "").lower()
        action = params.get("action", "restore").lower()
        
        if not target:
            return {"success": False, "message": "No target window specified."}
            
        import ctypes
        
        EnumWindows = ctypes.windll.user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        GetWindowText = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible
        ShowWindow = ctypes.windll.user32.ShowWindow
        SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow
        PostMessage = ctypes.windll.user32.PostMessageW
        
        SW_HIDE = 0
        SW_MAXIMIZE = 3
        SW_MINIMIZE = 6
        SW_RESTORE = 9
        WM_CLOSE = 0x0010
        
        found_hwnds = []
        
        def foreach_window(hwnd, lParam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLength(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    GetWindowText(hwnd, buff, length + 1)
                    if target in buff.value.lower():
                        found_hwnds.append(hwnd)
            return True
            
        EnumWindows(EnumWindowsProc(foreach_window), 0)
        
        if not found_hwnds:
            return {"success": False, "message": f"Could not find any window matching '{target}'."}
            
        hwnd = found_hwnds[0]
        
        try:
            if action == "minimize":
                ShowWindow(hwnd, SW_MINIMIZE)
            elif action == "maximize":
                ShowWindow(hwnd, SW_MAXIMIZE)
            elif action == "close":
                PostMessage(hwnd, WM_CLOSE, 0, 0)
            else:
                ShowWindow(hwnd, SW_RESTORE)
                SetForegroundWindow(hwnd)
                
            log_action("manage_window", {"target": target, "action": action, "hwnd": hwnd})
            return {"success": True, "message": f"Successfully performed '{action}' on window containing '{target}'."}
        except Exception as e:
            return {"success": False, "message": f"Failed to manage window: {e}"}

    def _run_python(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a dynamic python script via Subprocess for complex PyAutoGUI orchestration.
        Requires user confirmation via pipeline!
        """
        code = params.get("code", "")
        if not code:
            return {"success": False, "message": "No python code provided."}
            
        import tempfile
        import uuid
        
        temp_dir = Path(tempfile.gettempdir())
        script_path = temp_dir / f"jarvis_auto_{uuid.uuid4().hex}.py"
        
        try:
            script_path.write_text(code, encoding="utf-8")
            
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            out = (result.stdout or result.stderr or "").strip()[:1000]
            log_action("run_python", {"code_len": len(code)})
            
            if result.returncode == 0:
                return {"success": True, "message": f"Python script executed successfully. Output: {out}"}
            else:
                return {"success": False, "message": f"Python script failed: {out}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Python script timed out after 60 seconds."}
        except Exception as e:
            return {"success": False, "message": f"Error running script: {e}"}
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except:
                    pass

    # --- Hardware Control Methods ---

    def _mouse_move(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not pyautogui: return {"success": False, "message": "pyautogui not installed"}
        try:
            x, y = params.get("x", 0), params.get("y", 0)
            duration = params.get("duration", 0.2)
            relative = params.get("relative", False)
            if relative:
                pyautogui.move(x, y, duration=duration)
            else:
                pyautogui.moveTo(x, y, duration=duration)
            log_action("mouse_move", {"x": x, "y": y, "relative": relative})
            return {"success": True, "message": f"Moved mouse to ({x}, {y})"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _mouse_click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not pyautogui: return {"success": False, "message": "pyautogui not installed"}
        try:
            button = params.get("button", "left")
            clicks = params.get("clicks", 1)
            pyautogui.click(button=button, clicks=clicks)
            log_action("mouse_click", {"button": button, "clicks": clicks})
            return {"success": True, "message": f"Clicked mouse {clicks} times ({button})"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _keyboard_type(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not pyautogui: return {"success": False, "message": "pyautogui not installed"}
        try:
            text = params.get("text", "")
            interval = params.get("interval", 0.05)
            pyautogui.write(text, interval=interval)
            log_action("keyboard_type", {"text": text[:20]})
            return {"success": True, "message": "Typed text."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _keyboard_hotkey(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not pyautogui: return {"success": False, "message": "pyautogui not installed"}
        try:
            keys = params.get("keys", [])
            if isinstance(keys, str):
                keys = keys.split('+')
            keys = [k.strip() for k in keys if k.strip()]
            pyautogui.hotkey(*keys)
            log_action("keyboard_hotkey", {"keys": keys})
            return {"success": True, "message": f"Pressed hotkey {'+'.join(keys)}"}
        except Exception as e:
            return {"success": False, "message": str(e)}


def _apply_unified_diff_to_lines(original_lines: list[str], diff_text: str) -> Optional[list[str]]:
    """
    Minimal unified-diff applier for one file.
    Supports @@ hunk blocks. Requires exact match of ' ' and '-' lines.
    """
    lines = diff_text.splitlines()
    # Strip any leading file headers; keep hunks only.
    i = 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    if i >= len(lines):
        return None

    out = original_lines[:]
    out_offset = 0

    import re
    hunk_header = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")

    while i < len(lines):
        if not lines[i].startswith("@@"):
            i += 1
            continue

        m = hunk_header.match(lines[i])
        if not m:
            return None
        old_start = int(m.group(1))
        # old_count = int(m.group(2) or "1")
        i += 1

        # Convert 1-based to 0-based index
        out_idx = (old_start - 1) + out_offset

        # Collect hunk lines until next header or end
        hunk_lines: list[str] = []
        while i < len(lines) and not lines[i].startswith("@@"):
            hunk_lines.append(lines[i])
            i += 1

        # Apply hunk
        cur = out_idx
        new_segment: list[str] = []
        for hl in hunk_lines:
            if hl.startswith("\\"):
                # "\ No newline at end of file" — ignore
                continue
            if not hl:
                # Empty line in diff is still a context line with prefix missing -> invalid
                return None
            prefix = hl[0]
            text = hl[1:]
            if prefix == " ":
                # Context must match
                if cur >= len(out) or out[cur] != text:
                    return None
                new_segment.append(text)
                cur += 1
            elif prefix == "-":
                # Deletion must match
                if cur >= len(out) or out[cur] != text:
                    return None
                cur += 1
            elif prefix == "+":
                new_segment.append(text)
            else:
                return None

        # Replace the consumed portion [out_idx:cur] with new_segment
        out[out_idx:cur] = new_segment
        out_offset += len(new_segment) - (cur - out_idx)

    return out

