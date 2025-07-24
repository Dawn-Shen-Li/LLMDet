from transformers import AutoConfig, AutoTokenizer
from llava.model import LlavaQwenForCausalLM

model_id = "llava-qwen2-7b-si-hf"
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = LlavaQwenForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
model.eval()


from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

processor = LlavaOnevisionProcessor.from_pretrained(model_id)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    return_dict=True
).eval()

# Prepare input
img = Image.open("img.jpg").convert("RGB")
messages = [
  {"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe with CoT."}]}
]
inputs = processor(messages, images=[img], return_tensors="pt").to(model.device)

# Generate
out = model.generate(
    **inputs,
    max_new_tokens=256,
    output_hidden_states=True,
    return_dict_in_generate=True
)

cot = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
last_h = out.hidden_states[-1]  # final-layer hidden states

print("CoT:", cot)
print("Hidden states shape:", last_h.shape)  # (batch, seq_len, hidden_dim)