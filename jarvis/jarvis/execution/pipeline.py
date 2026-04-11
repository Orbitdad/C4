import logging
from typing import Dict, Any
from .executor import Executor
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)

class ExecutionPipeline:
    """
    Manages execution graphs.
    Instead of a flat list, tasks are nodes with 'on_success' and 'on_failure' edges.
    """
    def __init__(self, executor: Executor, planner=None):
        self.executor = executor
        self.planner = planner
        
    def run_graph(self, graph_data: Dict[str, Any]) -> str:
        nodes = graph_data.get("nodes", {})
        curr_node_id = graph_data.get("start_node")
        
        if not nodes or not curr_node_id:
            return "Empty execution pipeline."
            
        final_result = "Done"
        
        self._interrupted = False
        def _on_int(evt):
            self._interrupted = True
            
        bus.subscribe("system.interrupt", _on_int)
        
        try:
            while curr_node_id and curr_node_id in nodes:
                if self._interrupted:
                    logger.warning(f"[Pipeline] Execution aborted due to system interrupt at node {curr_node_id}")
                    return "Execution cancelled by system interrupt."
                    
                node = nodes[curr_node_id]
                logger.info(f"[Pipeline] Current Node: {curr_node_id}")
            
                # --- Safety Sandbox Injection ---
                if node.get("requires_confirmation"):
                    from .safety_layer import sandbox
                    logger.warning(f"[SafetyLayer] Node {curr_node_id} requires user verbal confirmation.")
                    auth = sandbox.require_auth(curr_node_id, node.get("type"))
                    if not auth:
                        logger.warning(f"[SafetyLayer] Node {curr_node_id} was DENIED. Aborting step.")
                        curr_node_id = node.get("on_failure")
                        continue
                # --------------------------------

                try:
                    # Execution
                    step_obj = {"type": node.get("type"), "params": node.get("params", {})}
                
                    # We natively print back to UI if we are telling the user
                    if node.get("type") == "tell_user":
                        final_result = node.get("params", {}).get("message", "Done")
                        res = {"success": True, "message": final_result}
                    else:
                        res = self.executor.execute_step(step_obj)
                        final_result = res.get("message", "Done")
                
                    if res.get("success", True):
                        curr_node_id = node.get("on_success")
                    else:
                        from jarvis.memory.decision_trace import decision_trace
                        logger.warning(f"[Pipeline] Node {curr_node_id} failed. Dropping to fallback edge.")
                    
                        raw_fallback = node.get("on_failure")
                        if raw_fallback == "replan" or not raw_fallback:
                            logger.info(f"[Pipeline] Initiating dynamic re-planning for failed node {curr_node_id}")
                            decision_trace.log_decision("PlannerAgent", "Mid-Execution Replan", f"Node '{curr_node_id}' failed. Branching to fallback generator.")
                        
                            if self.planner and getattr(self.planner, "llm", None):
                                error_msg = res.get("message", res.get("error", "Unknown error"))
                                action_desc = f"{node.get('type')} with {node.get('params')}"
                                prompt = f"""You are JARVIS's execution planner. A step in your execution graph failed.
    Failed Step: {action_desc}
    Error Received: {error_msg}

    Generate a NEW JSON execution graph to recover from this and achieve the user's likely intent. 
    Follow the exact same strict JSON schema as before (start_node, nodes map, requires_confirmation boolean, on_success, on_failure).
    If it is unrecoverable, return a single graph node telling the user it failed permanently.
    Return ONLY valid JSON.
    """
                                new_graph = self.planner.llm.generate(prompt)
                                try:
                                    import json
                                    cleaned = new_graph.strip()
                                    if cleaned.startswith("```json"): cleaned = cleaned[7:]
                                    if cleaned.startswith("```"): cleaned = cleaned[3:]
                                    cleaned = cleaned.strip("` \n")
                                    parsed_graph = json.loads(cleaned)
                                    if parsed_graph and "nodes" in parsed_graph:
                                        logger.info("[Pipeline] Dynamic replanning generated a new successful graph.")
                                        nodes = parsed_graph["nodes"]
                                        curr_node_id = parsed_graph.get("start_node")
                                    
                                        # Reinforcement Learning feedback hook (for the failure)
                                        bus.publish(SystemEvent(
                                            name="ai.learning.failure_caught",
                                            data={"failed_node": node, "error": error_msg},
                                            priority=EventPriority.NORMAL
                                        ))
                                        continue # Loop continues with new nodes and curr_node_id
                                except Exception as parse_err:
                                    logger.error(f"[Pipeline] Re-planning failed to generate valid graph: {parse_err}")
                                
                            curr_node_id = None
                            final_result = f"Failed due to {res.get('message', res.get('error', 'error'))}. I attempted to replan but could not recover."
                        else:
                            curr_node_id = raw_fallback
                    
                        # Reinforcement Learning feedback hook
                        bus.publish(SystemEvent(
                            name="ai.learning.failure_caught",
                            data={"failed_node": node, "error": res.get("error")},
                            priority=EventPriority.NORMAL
                        ))
                    
                except Exception as e:
                    logger.error(f"[Pipeline] Uncaught exception at {curr_node_id}: {e}")
                    curr_node_id = node.get("on_failure")
                    if not curr_node_id:
                        final_result = f"Pipeline crashed fatally: {e}"
        
        
            # If we reached the end successfully (no current node left but no fatal crash)            
            if not self._interrupted:
                bus.publish(SystemEvent(
                    name="ai.learning.success",
                    data={"intent": graph_data.get("start_node", "graph")},
                    priority=EventPriority.LOW
                ))
            return final_result
            
        finally:
            bus.unsubscribe("system.interrupt", _on_int)
