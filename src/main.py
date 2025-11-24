from QwenMem.modeling_qwen2_5_vl_with_memory import Qwen2_5_VLForConditionalGenerationWithMemory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = Qwen2_5_VLForConditionalGenerationWithMemory.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model.memory.load_state_dict(
    checkpoint_path="/Users/richardtang/Desktop/VIL/QwenMem/Qwen2_5_VL-3B-WithMemory/checkpoint.pth"
)

model.save_pretrained("/Users/richardtang/Desktop/VIL/QwenMem/src/temp/qwen2_5vl-3b-with-memory", safe_serialization=False)