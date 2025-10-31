import json
from tqdm import tqdm 
import os
import re

DATA_ROOT = "/projects/vig/Datasets/SpaceR-151k"

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def main():
    data = sorted(load_jsonl("SpaceR-151k.jsonl"), key=lambda x: x["path"])

    out_path = "spaceR_llama.jsonl"
    fout = open(out_path, "w")

    for item in tqdm(data, desc="Processing data"):
        # skip modalities we don't want
        if item["problem_type"] in ["OCR", "free-form", "regression"]:
            continue
        
        media_path = os.path.join(DATA_ROOT, item["path"].lstrip("/"))
        if not media_path.endswith(".mp4"):
            continue

        question = item["problem"].strip()
        answer = extract_answer(item["solution"])

        if item["problem_type"] == "multiple choice":
            options = item["options"]
            options_str = "\n".join([f"{option}" for i, option in enumerate(options)])

        record = {
            "messages": [
                {
                    "content": f"<video>{question}\n{options_str}", 
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
            ],
            "videos": [
                media_path
            ]
        }

        fout.write(json.dumps(record) + "\n")

    fout.close()
    print(f"✅ Finished. Saved to {out_path}")

if __name__ == "__main__":
    main()
