#!/usr/bin/env python3
"""
Simple script for continuous text conversation with Qwen2_5_VLForConditionalGenerationWithMemory.
Type /quit to exit.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoProcessor
from src.modeling_qwen2_5_vl_with_memory import Qwen2_5_VLForConditionalGenerationWithMemory


def main():
    # Default model path - can be overridden via command line argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else "RichardGTang/Qwen2_5_VL-3B-WithMemory"
    
    print(f"Loading model: {model_name}")
    print("This may take a while...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_fast=True
    )
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Qwen2_5_VLForConditionalGenerationWithMemory.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print("\n" + "="*60)
    print("Chat started! Type '/quit' to exit.")
    print("="*60 + "\n")
    
    # Conversation history
    messages = []
    first_turn = True
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == "/quit":
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        })
        
        # Format messages with chat template
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = processor(
            text=text_prompt,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                reset_memory=first_turn,  # Reset memory only on first turn
            )
        
        # Extract only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        
        # Decode response
        response_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(response_text)
        print()
        
        # Add assistant response to history
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })
        
        # After first turn, don't reset memory
        first_turn = False


if __name__ == "__main__":
    main()

