# Llama-3.2-1B-ComputerEngineeringLLM
A custom fine-tuned large language model based on Meta's LLaMA 3.2 1B, specialized for computer engineering applications. Fine-tuned using datasets like Wikitext-2-raw-v1 and computer science and computer engineering dataset.

The model was fine-tuned using LoRA (Low-Rank Adaptation) adapters

---

<div align="center">
  <a href="https://github.com/IrfanUruchi/Llama-3.2-1B-ComputerEngineeringLLM">
    <img src="https://img.shields.io/badge/🔗_GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM">
    <img src="https://img.shields.io/badge/🤗_HuggingFace-Model_Repo-FFD21F?style=for-the-badge" alt="HuggingFace">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Model_Size-1B_parameters-blue" alt="Model Size">
  <img src="https://img.shields.io/badge/Quantization-8bit-green" alt="Quantization">
  <img src="https://img.shields.io/badge/Adapter-LoRA-orange" alt="Adapter">
  <img src="https://img.shields.io/badge/Context-8k-lightgrey" alt="Context">
</div>

---

# Model Details:

**Base Model:** Meta’s LLaMA 3.2 1B  
**Architecture:** LlamaForCausalLM  
- **Hidden Size:** 2048  
- **Number of Layers:** 16  
- **Number of Attention Heads:** 32
- **Quantization:** Loaded in 8-bit mode using BitsAndBytes  
- **Tokenizer:** Uses a vocabulary of 128256 tokens
- **Fine-Tuning Method:** LoRA(Low-Rank Adaptation)


# Usage Instructions:

### Option 1: From Hugging Face Hub (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto", 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False  # Required for proper Llama tokenization
)

prompt = "Explain the von Neumann architecture:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  

outputs = model.generate(
    **inputs,
    max_new_tokens=200,  
    temperature=0.7,     
    top_p=0.9,          
    do_sample=True,   
    repetition_penalty=1.1  
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: Local Installation (Git LFS Required)


```python

from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your local path
model_path = "./Llama-3.2-1B-ComputerEngineeringLLM"  

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,  # Required for Llama tokenizer
    local_files_only=True
)
```

*Recomended Config*

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7, 
    top_p=0.9,     
    do_sample=True,
    repetition_penalty=1.1  
)
```


# License and Attribution:

This model is a derivative work based on Meta’s LLaMA 3.2 1B and is distributed under the LLaMA 3.2 Community License.
Please see the LICENSE file for the full text of the license.


# Attribution:

“Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Built with Llama.”
For more information on the base model, please visit :

https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE

# Known Limitations:

The model is specialized for computer engineering topics and may not work as well on unrelated subjects, some outputs may require further prompt engineering for optimal results. And occasional repetition or other artifacts may be present due to fine-tuning constraints.


---

## Citation


If using for academic research, please cite:

```bibtex
@misc{llama3.2-1b-eng-2025,
  title = {Llama-3.2-1B-Computer-Engineering-LLM},
  author = {Irfanuruchi},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM},
}
```







