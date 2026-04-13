import time
import queue
import threading
import uuid
from typing import Any, Callable, Dict, Optional
import logging
from .executor import Executor
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)

class ActionScheduler:
    """
    Priority-based task scheduler.
    Listens for execution requests and runs them based on priority.
    """
    def __init__(self, executor: Executor):
        self.executor = executor
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.is_running = False
        self._thread = None
        self._cancelled_ids = set()
        self._cancelled_tokens = set()
        
        bus.subscribe("action.schedule", self._handle_schedule_request)
        bus.subscribe("action.cancel", self._handle_cancel_request)
        bus.subscribe("system.interrupt", self._on_interrupt)
        
    def _handle_schedule_request(self, event: SystemEvent):
        delay_seconds = event.data.get("delay_seconds", 0)
        
        if delay_seconds > 0:
            logger.info(f"[Scheduler] Delaying task by {delay_seconds}s (Priority {event.priority.name})")
            timer = threading.Timer(delay_seconds, self._enqueue_delayed, args=[event])
            timer.daemon = True
            timer.start()
        else:
            self._enqueue_delayed(event)
            
    def _enqueue_delayed(self, event: SystemEvent):
        logger.debug(f"[Scheduler] Queued task at priority {event.priority.name}")
        payload = dict(event.data or {})
        payload.setdefault("task_id", str(uuid.uuid4()))
        payload.setdefault("retries", 0)
        payload.setdefault("attempt", 0)
        payload.setdefault("retry_delay_seconds", 0.0)
        self.task_queue.put((-event.priority.value, event.timestamp, payload))

    def _handle_cancel_request(self, event: SystemEvent):
        task_id = str((event.data or {}).get("task_id") or "").strip()
        token = str((event.data or {}).get("cancel_token") or "").strip()
        if task_id:
            self._cancelled_ids.add(task_id)
        if token:
            self._cancelled_tokens.add(token)
        if task_id or token:
            logger.info(f"[Scheduler] Cancellation recorded task_id={task_id or '-'} token={token or '-'}")
        
    def _on_interrupt(self, event: SystemEvent):
        logger.warning("[Scheduler] Interrupt received, clearing pending execution queue.")
        with self.task_queue.mutex:
            self.task_queue.queue.clear()
        self._cancelled_tokens.add("interrupt")
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[Scheduler] Priority Action Scheduler Online.")
        
    def stop(self):
        self.is_running = False
        self.task_queue.put((-999, time.time(), {"step": None}))
        if self._thread:
            self._thread.join()
            
    def _run_loop(self):
        while self.is_running:
            try:
                priority, ts, task_data = self.task_queue.get(timeout=1.0)
                step = task_data.get("step")
                task_id = str(task_data.get("task_id", "")).strip()
                cancel_token = str(task_data.get("cancel_token", "")).strip()
                if task_id and task_id in self._cancelled_ids:
                    self.task_queue.task_done()
                    continue
                if cancel_token and cancel_token in self._cancelled_tokens:
                    self.task_queue.task_done()
                    continue
                
                if not step:
                    self.task_queue.task_done()
                    continue
                    
                logger.info(f"[Scheduler] Executing Task (Priority {-priority})")
                try:
                    result = self.executor.execute_step(step)
                    
                    callback = task_data.get("callback")
                    if callback:
                        callback(result)
                        
                    bus.publish(SystemEvent(
                        name="action.completed",
                        data={"step": step, "result": result},
                        priority=EventPriority.NORMAL
                    ))
                    
                except Exception as e:
                    logger.error(f"[Scheduler] Execution failed: {e}")
                    result = {"success": False, "message": str(e), "error": str(e)}
                    retries = int(task_data.get("retries", 0) or 0)
                    attempt = int(task_data.get("attempt", 0) or 0)
                    if attempt < retries:
                        retry_payload = dict(task_data)
                        retry_payload["attempt"] = attempt + 1
                        retry_delay = float(task_data.get("retry_delay_seconds", 0.0) or 0.0)
                        if retry_delay > 0:
                            timer = threading.Timer(
                                retry_delay,
                                self.task_queue.put,
                                args=[(-priority, time.time(), retry_payload)],
                            )
                            timer.daemon = True
                            timer.start()
                        else:
                            self.task_queue.put((-priority, time.time(), retry_payload))
                    else:
                        bus.publish(SystemEvent(
                            name="action.failed",
                            data={"step": step, "error": str(e), "task_id": task_id},
                            priority=EventPriority.HIGH
                        ))
                if not result.get("success", False):
                    retries = int(task_data.get("retries", 0) or 0)
                    attempt = int(task_data.get("attempt", 0) or 0)
                    if attempt < retries:
                        retry_payload = dict(task_data)
                        retry_payload["attempt"] = attempt + 1
                        retry_delay = float(task_data.get("retry_delay_seconds", 0.0) or 0.0)
                        if retry_delay > 0:
                            timer = threading.Timer(
                                retry_delay,
                                self.task_queue.put,
                                args=[(-priority, time.time(), retry_payload)],
                            )
                            timer.daemon = True
                            timer.start()
                        else:
                            self.task_queue.put((-priority, time.time(), retry_payload))
                    else:
                        bus.publish(SystemEvent(
                            name="action.failed",
                            data={"step": step, "error": result.get("message", "execution failed"), "task_id": task_id, "result": result},
                            priority=EventPriority.HIGH
                        ))
                    
                self.task_queue.task_done()
            except queue.Empty:
                pass
