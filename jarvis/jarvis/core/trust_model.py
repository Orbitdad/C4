from enum import IntEnum

class TrustLevel(IntEnum):
    READ_ONLY = 0    # Gathering context, reading state
    SAFE = 1         # Opening apps, minor UI nav
    REVERSIBLE = 2   # Creating files, edits
    DESTRUCTIVE = 3  # Deletions, shutdowns

class TrustHierarchy:
    @staticmethod
    def get_action_level(action_type: str) -> TrustLevel:
        """Map action types to their corresponding trust levels."""
        mapping = {
            "read_file": TrustLevel.READ_ONLY,
            "tell_time": TrustLevel.READ_ONLY,
            "tell_date": TrustLevel.READ_ONLY,
            "web_search": TrustLevel.READ_ONLY,
            
            "open_app": TrustLevel.SAFE,
            "open_url": TrustLevel.SAFE,
            "play_media": TrustLevel.SAFE,
            "mouse_move": TrustLevel.SAFE,
            
            "create_file": TrustLevel.REVERSIBLE,
            "modify_file": TrustLevel.REVERSIBLE,
            "mouse_click": TrustLevel.REVERSIBLE,
            "keyboard_type": TrustLevel.REVERSIBLE,
            
            "delete_file": TrustLevel.DESTRUCTIVE,
            "run_command": TrustLevel.DESTRUCTIVE,
            "sys.shutdown": TrustLevel.DESTRUCTIVE,
            "keyboard_hotkey": TrustLevel.DESTRUCTIVE,
        }
        return mapping.get(action_type, TrustLevel.DESTRUCTIVE)  # Default to highest if unknown
