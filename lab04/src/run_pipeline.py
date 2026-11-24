import subprocess
import sys

steps = [
    "lab04/src/ingest.py",
    "lab04/src/validate.py",
    "lab04/src/clean.py",
    "lab04/src/features.py",
    "lab04/src/label_quality.py",
    "lab04/src/hash_data.py",
]

for step in steps:
    print(f"\n=== Running {step} ===", flush=True)
    subprocess.run([sys.executable, step], check=True)

print("\nPipeline finished.")