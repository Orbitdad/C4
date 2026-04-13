import logging
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)

class ErrorRecoveryLayer:
    """
    Handles exceptions, retries, and rollbacks for actions to prevent
    the system from crashing due to single misinterpretations.
    """
    
    @staticmethod
    def execute_with_retry(func: Callable, max_retries: int = 3, backoff_factor: float = 1.5, *args, **kwargs) -> Any:
        retries = 0
        delay = 1.0
        
        while retries < max_retries:
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict) and not result.get("success", True):
                    raise Exception(result.get("message", "Action failed"))
                return result
            except Exception as e:
                retries += 1
                logger.warning(f"[ErrorRecovery] Action failed (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    logger.error("[ErrorRecovery] Max retries reached. Action aborted.")
                    return {"success": False, "message": str(e)}
                time.sleep(delay)
                delay *= backoff_factor
                
    @staticmethod
    def check_confidence(confidence: float, threshold: float = 0.7) -> bool:
        """Validate LLM or Vision confidence before acting."""
        if confidence < threshold:
            logger.warning(f"[ErrorRecovery] Confidence {confidence} below threshold {threshold}. Discarding.")
            return False
        return True
