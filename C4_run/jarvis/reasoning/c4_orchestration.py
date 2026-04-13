from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import threading
import time
import uuid

from jarvis.core.event_bus import bus, EventPriority, SystemEvent


@dataclass
class JobStep:
    name: str
    status: str = "pending"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodingJob:
    job_id: str
    request: str
    status: str = "planned"
    steps: List[JobStep] = field(default_factory=list)
    current_step: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    telemetry: List[Dict[str, Any]] = field(default_factory=list)
    error_fingerprints: set[str] = field(default_factory=set)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, CodingJob] = {}
        self._last_job_id: Optional[str] = None

    def create(self, request: str) -> CodingJob:
        with self._lock:
            job = CodingJob(job_id=str(uuid.uuid4()), request=request)
            self._jobs[job.job_id] = job
            self._last_job_id = job.job_id
            return job

    def get(self, job_id: str) -> Optional[CodingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def last(self) -> Optional[CodingJob]:
        with self._lock:
            if not self._last_job_id:
                return None
            return self._jobs.get(self._last_job_id)

    def update_status(self, job_id: str, status: str, current_step: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            if current_step is not None:
                job.current_step = current_step
            job.updated_at = time.time()

    def add_step(self, job_id: str, name: str, status: str = "pending", meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.steps.append(JobStep(name=name, status=status, meta=meta or {}))
            job.updated_at = time.time()

    def add_telemetry(self, job_id: str, row: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.telemetry.append(row)
            job.updated_at = time.time()

    def add_error_fingerprint(self, job_id: str, fingerprint: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if fingerprint in job.error_fingerprints:
                return False
            job.error_fingerprints.add(fingerprint)
            job.updated_at = time.time()
            return True


class StageTimer:
    def __init__(self, store: InMemoryJobStore, job_id: str, stage: str, model: str, tokens: Optional[int] = None) -> None:
        self.store = store
        self.job_id = job_id
        self.stage = stage
        self.model = model
        self.tokens = tokens
        self.started_at = time.time()

    def close(self) -> None:
        self.store.add_telemetry(
            self.job_id,
            {
                "stage": self.stage,
                "model": self.model,
                "tokens": self.tokens,
                "elapsed_ms": int((time.time() - self.started_at) * 1000),
            },
        )


def map_command_priority(command_text: str) -> EventPriority:
    c = (command_text or "").lower()
    if "build" in c:
        return EventPriority.CRITICAL
    if "install" in c:
        return EventPriority.HIGH
    if "dev" in c or "start" in c or "serve" in c:
        return EventPriority.NORMAL
    return EventPriority.LOW


def fingerprint_errors(errors: List[Dict[str, Any]]) -> str:
    payload = []
    for e in errors:
        payload.append(
            {
                "file": e.get("file", ""),
                "line": e.get("line", 0),
                "error_type": e.get("error_type", ""),
                "message": (e.get("message", "") or "")[:200],
            }
        )
    return hashlib.sha256(str(sorted(payload, key=lambda x: (x["file"], x["line"], x["error_type"], x["message"]))).encode("utf-8")).hexdigest()


def schedule_command_and_wait(
    command: str,
    cwd: str,
    retries: int = 0,
    delay_seconds: float = 0.0,
    priority: Optional[EventPriority] = None,
    cancel_token: str = "",
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 90.0,
) -> Dict[str, Any]:
    done = threading.Event()
    out: Dict[str, Any] = {"success": False, "message": "No result."}
    task_id = str(uuid.uuid4())
    step = {"type": "run_command", "command": command, "cwd": cwd}

    def _cb(result: Dict[str, Any]) -> None:
        out.update(result or {})
        done.set()

    event_priority = priority or map_command_priority(command)
    bus.publish(
        SystemEvent(
            name="action.schedule",
            priority=event_priority,
            data={
                "task_id": task_id,
                "cancel_token": cancel_token,
                "delay_seconds": max(0.0, float(delay_seconds or 0.0)),
                "retries": max(0, int(retries or 0)),
                "retry_delay_seconds": max(0.0, float(retry_delay_seconds or 0.0)),
                "step": step,
                "callback": _cb,
            },
        )
    )
    done.wait(timeout=timeout_seconds)
    if not done.is_set():
        return {"success": False, "message": f"Command timeout: {command}", "command": command}
    return out
