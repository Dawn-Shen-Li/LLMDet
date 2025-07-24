#!/bin/bash
# https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md
# Name your conda environment
ENV_NAME="llmdet"

# Create the conda environment with Python 3.10 for best compatibility
conda create -n $ENV_NAME python=3.11 -y

# Activate the environment
conda init
conda activate $ENV_NAME

# Install PyTorch 2.2.1 with CUDA 12.1 from PyTorch channel
# Also installs compatible torchvision and torchaudio
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install transformers==4.45.2 nltk peft wandb jsonlines
pip install numpy==1.23.2 scipy==1.9.3 opencv-python==4.7.0.72

# MM-related packages
pip install -U openmim
mim install mmcv==2.2.0 mmengine==0.10.5

# Vision/Detection utilities
pip install timm pycocotools shapely terminaltables scipy

# Large model & optimization tools
pip install deepspeed fairscale lvis 
pip install git+https://github.com/lvis-dataset/lvis-api.git"

# 请注意由于 LVIS 第三方库暂时不支持 numpy 1.24，因此请确保您的 numpy 版本符合要求。建议安装 numpy 1.23 版本。
# LVIS 需要 spacy 注意其版本和 deepspeed 版本的兼容性cho 

✅ Conda environment '$ENV_NAME' created and ready."

## for SAIL
pip install av decord deepspeed fvcore huggingface_hub natsort matplotlib openai pycocoevalcap python_magic thop shortuuid
pip install datasets==2.17.0 bitsandbytes==0.42.0

# Llava 
# LlamaModel : Base class combining vision tower, resampler, and projector (but no language modeling head).
# └── LlamaForCausalLM : Adds LM head and generation logic to a MetaModel, providing generate() support 

# LlavaMetaModel (vision + LLM backbone) 
# ├── LlavaQwenModel
# ├── LlavaMptModel
# ├── LlavaMistralModel
# ├── LlavaGemmaModel
# ├── LlavaMixtralModel
# └── LlavaLlamaModel
 
# └── LlavaQwenModel
#     └── LlavaQwenForCausalLM



# Huggingface wrapper for LlavaOnevision
# PreTrainedModel
# └── LlavaOnevisionPreTrainedModel : Abstract HF base class for loading & configuring this family of models.
#     └── LlavaOnevisionModel: Vision + language backbone without generation capabilities.
#        └── LlavaOnevisionForConditionalGeneration:Extends the backbone with a causal LM head, giving .generate() for vision-conditioned text generation.

