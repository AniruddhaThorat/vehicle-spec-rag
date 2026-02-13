from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def extract_specs(context, query):

    prompt = f"""
You are an automotive specification extraction engine.

RULES:
1. Return ONLY valid JSON.
2. Use EXACT schema below.
3. DO NOT explain.
4. DO NOT infer.
5. If spec not present in context then return [].

Schema:
[
  {{
    "component": "",
    "spec_type": "",
    "value": "",
    "unit": ""
  }}
]

Context:
{context}

Query:
{query}

JSON:
"""

    output = generator(
        prompt,
        max_new_tokens=150,
        return_full_text=False,
        do_sample=False,
        temperature=0
    )

    return output[0]["generated_text"].strip()