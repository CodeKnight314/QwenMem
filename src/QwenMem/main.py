from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
messages = [
    {"role": "user", "content": r"{{content}}"},
    {"role": "assistant", "content": r"{{content}}"},
    {"role": "system", "content": r"{{content}}"},
    {"role": "tool", "content": r"{{content}}"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)