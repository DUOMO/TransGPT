import argparse
from finetune_visualglm import FineTuneVisualGLMModel
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
from model import chat, VisualGLMModel
#model, model_args = VisualGLMModel.from_pretrained('visualglm-6b', args=argparse.Namespace(fp16=True, skip_init=True))
from sat.model import AutoModel
model, model_args = AutoModel.from_pretrained(
        'merge_lora_p7_54000',
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
    ))
model = model.eval()

from sat.model.mixins import CachedAutoregressiveMixin
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
image_path = "examples/bz1.png"
response, history, cache_image = chat(image_path, model, tokenizer, "图中的标志表示什么含义？", history=[])
print(response)
