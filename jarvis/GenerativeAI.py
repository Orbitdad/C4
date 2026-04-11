# pyre-ignore-all-errors
import os
import json
import ollama
import sys

# Ensure jarvis directory is in path for imports
jarvis_dir = os.path.dirname(os.path.abspath(__file__))
if jarvis_dir not in sys.path:
    sys.path.insert(0, jarvis_dir)

try:
    from memory import get_recent_memories
except ImportError:
    # Fallback for when run from parent directory
    try:
        from jarvis.memory import get_recent_memories
    except ImportError:
        def get_recent_memories(limit=10): return []

# System prompt directing Ollama to return parsed JSON
SYS_PROMPT = """
You are the reasoning engine for an autonomous voice assistant called C4.
The user is going to give you a voice command that C4 did not understand natively.
Your job is to determine what the user wants to do, and output a strict JSON strictly matching one of the following schemas:

1. Conversational / Q&A:
{"action": "speak", "text": "A brief, natural conversational response to the user."}
Use this when the user is asking a general question, saying hello, or making small talk. Keep the text concise and natural for TTS.

2. Open Application / Website:
{"action": "open", "target": "notepad"}
Use this when the user says "open [app name]" or "launch [website]". Provide the name of the app or website as 'target'.

3. System Command:
{"action": "command", "cmd": "dir", "description": "Listing files in the current directory"}
Use this when the user explicitly wants to run a shell/system command (e.g. "what files are in this directory" -> "dir", or "close the active window" -> "taskkill /f /im [app].exe"). Always provide a natural, 'Jarvis-like' description of what the command does.

Return ONLY the raw JSON. Do not include markdown formatting like ```json.
"""

def parse_command_with_llm(query: str):
    """
    Takes an unrecognized voice query, sends it to Ollama (llama3), 
    and returns a parsed Python dictionary of the action.
    """
    try:
        memories = get_recent_memories(limit=10)
        messages = [
            {'role': 'system', 'content': SYS_PROMPT},
        ]
        
        # Add conversation context
        for mem in memories:
            messages.append({'role': mem['role'], 'content': mem['content']})
            
        # Add current user query
        messages.append({'role': 'user', 'content': query})
        
        # Use Client with a real timeout (15 seconds)
        client = ollama.Client(host='http://localhost:11434', timeout=15.0)
        response = client.chat(
            model='llama3.2',
            messages=messages,
            format='json',
            options={
                'temperature': 0.0,
            }
        )
        
        result_text = response['message']['content'].strip()
        result_dict = json.loads(result_text)
        return result_dict
    except json.JSONDecodeError as e:
        print(f"[Error] JSON Decode failed: {e}")
        return {"action": "speak", "text": "I encountered an error parsing my own thoughts."}
    except Exception as e:
        error_msg = str(e)
        print(f"[Error] GenerativeAI Failure: {error_msg}")
        if "connection" in error_msg.lower() or "11434" in error_msg:
            return {"action": "speak", "text": "I cannot connect to Ollama. Please ensure it is running by executing 'ollama serve' in a terminal."}
        if "timeout" in error_msg.lower():
             return {"action": "speak", "text": "The request to my cognitive engine timed out. Ollama might be overloaded or unresponsive."}
        return {"action": "speak", "text": f"I encountered an error connecting to my cognitive engine: {error_msg}. Please ensure Ollama is running and the 'llama3' model is downloaded."}
