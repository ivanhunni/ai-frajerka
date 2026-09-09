# evaluation/evaluate_personality.py
import json
import glob
import os
from inference.engine import InferenceEngine

def evaluate_model():
    engine = InferenceEngine()
    test_files = glob.glob("dataset/evaluation/**/*.json", recursive=True)
    
    total_tests = 0
    print("=== Spúšťam testovanie osobnosti a konverzácie ===")

    for test_file in test_files:
        print(f"\nTestujem súbor: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            tests = data if isinstance(data, list) else [data]

            for test in tests:
                total_tests += 1
                prompt = test.get("instruction", test.get("prompt", "Ahoj!"))
                
                messages = [{"role": "user", "content": prompt}]
                reply = engine.generate_reply(messages)

                print(f"[{total_tests}] Otázka: {prompt}")
                print(f"     Odpoveď AI: {reply}")
                print("-" * 40)

    print(f"\nVyhodnotenie dokončené. Celkovo spustených testov: {total_tests}")

if __name__ == "__main__":
    evaluate_model()