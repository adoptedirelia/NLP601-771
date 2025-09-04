#!/usr/bin/env python3
"""
Simple Perplexity Analysis + Sampling Comparison
"""
import math
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_perplexity(text, model, tokenizer, device):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

def generate(prompt, model, tokenizer, device, temperature=None, length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": length, "pad_token_id": tokenizer.eos_token_id}
    if temperature is None or temperature == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    model_name = "distilbert/distilgpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    paragraph = "That is my spot. In an ever-changing world, it is a single point of consistency. If my life were expressed as a function on a four-dimensional Cartesian coordinate system, that spot, at the moment I first sat on it, would be 0-0-0-0."
    words = paragraph.split(' ')
    random.shuffle(words)
    shuffled = ' '.join(words)
    ppl_orig = compute_perplexity(paragraph, model, tokenizer, device)
    ppl_shuf = compute_perplexity(shuffled, model, tokenizer, device)
    print("Original PPL:", ppl_orig)
    print("Shuffled PPL:", ppl_shuf)


    prompt = "Once upon a time"
    temps = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
    for T in temps:
        out = generate(prompt, model, tokenizer, device, temperature=T, length=500)
        print(f"\n--- Temperature={T} ---\n")
        print(out)

if __name__ == "__main__":
    main()
