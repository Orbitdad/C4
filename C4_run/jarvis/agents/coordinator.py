import logging
from typing import Optional, Any
import json
from jarvis.core.world_state import world

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """
    The Meta-Controller for JARVIS. 
    Reads the WorldState and delegates user queries to specialized Sub-Agents.
    """
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.agents = {
            "VisionAgent": "Handles queries about what the user is looking at or their physical environment.",
            "SystemAgent": "Handles OS commands, file manipulation, and changing system states.",
            "KnowledgeAgent": "Handles general knowledge questions, math, and conversational reasoning.",
            "PlannerAgent": "Handles complex multi-step workflows like deployments or research passes."
        }
        
    def route_query(self, query: str) -> str:
        state = world.get_snapshot()
        prompt = f"""
You are the JARVIS Meta-Coordinator.
Current World State:
- Active Window: {state['user_environment'].get('active_window')}
- Inferred Activity: {state['user_environment'].get('inferred_activity')}
- System Load: CPU {state['system_health'].get('cpu_percent')}%
        
User Query: "{query}"
        
Available Sub-Agents:
{json.dumps(self.agents, indent=2)}
        
Which agent is best suited to handle this request? Return ONLY the exact name of the agent.
"""
        if not self.llm:
            return "KnowledgeAgent"
            
        try:
            response = self.llm.generate(prompt).strip()
            for agent in self.agents:
                if agent.lower() in response.lower():
                    return agent
        except Exception as e:
            logger.error(f"[Coordinator] Routing failure: {e}")
            
        return "KnowledgeAgent"

    def execute(self, query: str) -> str:
        """
        In a microservice architecture, this would publish to the EventBus.
        For monolithic phase, it returns the delegated agent.
        """
        agent_name = self.route_query(query)
        world.add_history(f"Coordinator routed '{query}' to {agent_name}")
        logger.info(f"[Coordinator] Delegating query to {agent_name}")
        return agent_name
