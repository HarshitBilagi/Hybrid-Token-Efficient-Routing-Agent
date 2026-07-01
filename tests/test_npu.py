import openvino_genai as ov_genai
import time
import sys
sys.path.insert(0, ".")

from src.local_inference.phi_pipeline import PhiNPUPipeline

pipeline = PhiNPUPipeline()
pipeline.load()

result = pipeline.generate("What is the capital of France?", max_new_tokens=80)
print(result)
