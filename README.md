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
  - **BOS Token:** `<|begin_of_text|>`  
  - **EOS Token:** `<|end_of_text|>`  
  - **PAD Token:** `<|end_of_text|>`


# Usage Instructions:

To load and use the model with the Hugging Face Transformers library, you can use the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("", use_fast=False)

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

The model is specialized for computer engineering topics and may not perform as well on unrelated subjects.
Some outputs may exhibit occasional repetition or require further prompt engineering for optimal results.






