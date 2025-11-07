from transformers import AutoConfig, AutoProcessor
from QwenMem.modeling_qwen2_5_vl_with_memory import Qwen2_5_VLForConditionalGenerationWithMemory

def register_qwen_mem():
    AutoConfig.register_for_auto_class(
        "qwen2_5_vl_with_memory",
        Qwen2_5_VLForConditionalGenerationWithMemory
    )
    
    AutoConfig.model_type = "qwen2_5_vl_with_memory"
    Qwen2_5_VLForConditionalGenerationWithMemory.config_class.model_type = "qwen2_5_vl_with_memory"
    Qwen2_5_VLForConditionalGenerationWithMemory.config_class.architectures = [
        "Qwen2_5_VLForConditionalGenerationWithMemory"
    ]
    Qwen2_5_VLForConditionalGenerationWithMemory._auto_class = "AutoModelForVision2Seq"

    print("✅ Registered QwenMem (Qwen2.5-VL with VGGT+Memory) for LlamaFactory.")