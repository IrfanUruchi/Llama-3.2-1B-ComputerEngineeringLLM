# Llama-3.2-1B-ComputerEngineeringLLM
A custom fine-tuned large language model based on Meta's LLaMA 3.2 1B, specialized for computer engineering applications. Fine-tuned using datasets like Wikitext-2-raw-v1 and computer science and computer engineering dataset.

The model was fine-tuned using LoRA (Low-Rank Adaptation) adapters

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

To load and use the model with the git LFS, you can use the following code:

First for installing in your local machine first make sure you have git LFS installed and then(tutorial is using terminal) its recommended to install directly the files (especially the model.safetensors as it won't install automatically) one by one from the main repository if you dont have a git LFS subscription:

```python
pip install -U bitsandbytes
pip install transformers torch accelerate

git clone https://github.com/IrfanUruchi/Llama-3.2-1B-ComputerEngineeringLLM.git
cd Llama-3.2-1B-ComputerEngineeringLLM

git lfs pull

ls -lh

git lfs pull
```

After that you have first to localize the model in your computer :

```python

local_path = "put here your actual path for model"

 #example
local_path = "./Llama-3.2-1B-ComputerEngineeringLLM"  # if your have the file in the current directory
```

Then after that you can use the model:

```python

local_path = "./Llama-3.2-1B-ComputerEngineeringLLM"  # if your have the file in the current directory
model = AutoModelForCausalLM.from_pretrained("/content/Llama-3.2-1B-ComputerEngineeringLLM", device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/content/Llama-3.2-1B-ComputerEngineeringLLM", use_fast=False, local_files_only=True)


#The prompt
prompt = "Explain how computers process data."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.92)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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






