import sys
sys.path.insert(0, ".")

from src.remote_inference.remote_client import get_remote_client

client = get_remote_client()
result = client.generate("What is the capital of France?", max_new_tokens=80)
print(result)