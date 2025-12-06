#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer Comparison Tutorial
===============================

Author: Dr Giorgio Roffo
Year: 2025-2026

This tutorial demonstrates how different language models tokenize text differently.
We compare tokenizers from four popular models:
- Llama 3.1 (Meta)
- Phi-4 (Microsoft)
- Deepseek
- Qwen

You'll learn:
- How to load tokenizers from different models
- How to encode text into tokens
- How different tokenizers handle the same text
- Special tokens used by each model
- Tokenizer class types and vocabulary sizes

References:
-----------
For foundational work on attention mechanisms and transformers:

[1] Roffo, G. (2025). Timeline of Key Developments in Affinity-Based Attention.
    arXiv preprint. https://arxiv.org/abs/[to be added]

[2] Roffo, G. (2025). A Survey of Large Language Models: Foundations and Future Directions.
    arXiv preprint. https://arxiv.org/abs/[to be added]

[3] Roffo, G. (2025). The Origin of Self-Attention: Pairwise Affinity Matrices in Feature 
    Selection and the Emergence of Self-Attention. arXiv preprint arXiv:2507.14560.

[4] Roffo, G. (2024). Exploring Advanced Large Language Models with LLMSuite.
    arXiv preprint arXiv:2407.12036.

For the models used:
- Llama 3.1: Meta AI (2024). Llama 3.1 Model Card.
- Phi-4: Microsoft (2024). Phi-4 Technical Report.
- Deepseek: DeepSeek AI (2024). DeepSeek-V2 Technical Report.
- Qwen: Qwen Team (2024). Qwen2.5 Technical Report.
"""

import os  # For reading environment variables
from dotenv import load_dotenv  # For loading .env file
from transformers import AutoTokenizer  # For loading tokenizer

# Load environment variables from .env file
load_dotenv()  # Load .env file

# Get token from environment variable
hf_token = os.getenv("HF_TOKEN")  # Read HF_TOKEN from environment

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=hf_token)  # Load tokenizer

# Test: encode some text
text = "Hello, my name is Giorgio Roffo, who are you?"  # Text to encode
tokens = tokenizer.encode(text, return_tensors="pt")  # Convert text to tokens

# Get token IDs (convert tensor to list)
token_ids = tokens[0].tolist()  # Convert tensor to list of IDs

# Get token strings (convert IDs back to token strings)
token_strings = tokenizer.convert_ids_to_tokens(token_ids)  # Get token strings

# Print tokens and IDs in a list
print("Token ID | Token (raw) | Decoded as")
print("-" * 70)
for token_id, token_str in zip(token_ids, token_strings):  # Loop through both lists
    decoded_token = tokenizer.decode([token_id])  # Decode single token to see what it becomes
    print(f"{token_id:8} | {token_str:12} | '{decoded_token}'")  # Print ID, raw token, and decoded version

# Test: decode back to text
decoded = tokenizer.decode(tokens[0])  # Convert tokens back to text
print(f"\nDecoded text: {decoded}")  # Print the decoded text

print(f'\nVocabulary size (Llama 3.1): {len(tokenizer.vocab)}')  # Print vocabulary size

# Comparing the different tokenizers
# ===================================
print("\n" + "=" * 80)
print("Comparing Tokenizers: Llama 3.1 vs Phi-4 vs Deepseek vs Qwen")
print("=" * 80)

# Load other tokenizers
print("\nLoading tokenizers...")
tokenizer_phi4 = AutoTokenizer.from_pretrained("microsoft/phi-4", token=hf_token)  # Load Phi-4 tokenizer
tokenizer_deepseek = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2.5", token=hf_token)  # Load Deepseek tokenizer
tokenizer_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", token=hf_token)  # Load Qwen tokenizer
print("✓ All tokenizers loaded!")

# Print tokenizer information for each model
print("\n" + "=" * 100)
print("Tokenizer Information")
print("=" * 100)
print(f"Llama 3.1:  {type(tokenizer).__name__}")  # Print tokenizer class name
print(f"Phi-4:      {type(tokenizer_phi4).__name__}")  # Print tokenizer class name
print(f"Deepseek:   {type(tokenizer_deepseek).__name__}")  # Print tokenizer class name
print(f"Qwen:       {type(tokenizer_qwen).__name__}")  # Print tokenizer class name
print("=" * 100)

# Print special tokens for each tokenizer
print("\nSpecial Tokens")
print("=" * 100)
print(f"\nLlama 3.1 special tokens:")
if hasattr(tokenizer, 'special_tokens_map'):
    for key, value in tokenizer.special_tokens_map.items():  # Loop through special tokens
        print(f"  {key}: {value}")  # Print special token name and value

print(f"\nPhi-4 special tokens:")
if hasattr(tokenizer_phi4, 'special_tokens_map'):
    for key, value in tokenizer_phi4.special_tokens_map.items():  # Loop through special tokens
        print(f"  {key}: {value}")  # Print special token name and value

print(f"\nDeepseek special tokens:")
if hasattr(tokenizer_deepseek, 'special_tokens_map'):
    for key, value in tokenizer_deepseek.special_tokens_map.items():  # Loop through special tokens
        print(f"  {key}: {value}")  # Print special token name and value

print(f"\nQwen special tokens:")
if hasattr(tokenizer_qwen, 'special_tokens_map'):
    for key, value in tokenizer_qwen.special_tokens_map.items():  # Loop through special tokens
        print(f"  {key}: {value}")  # Print special token name and value
print("=" * 100)

# Text to tokenize
text = "Hello, my name is Giorgio Roffo, who are you?"  # Same text for all models

# Tokenize with each model
tokens_llama = tokenizer.encode(text, return_tensors="pt")[0].tolist()  # Llama tokens
tokens_phi4 = tokenizer_phi4.encode(text, return_tensors="pt")[0].tolist()  # Phi-4 tokens
tokens_deepseek = tokenizer_deepseek.encode(text, return_tensors="pt")[0].tolist()  # Deepseek tokens
tokens_qwen = tokenizer_qwen.encode(text, return_tensors="pt")[0].tolist()  # Qwen tokens

# Get token strings for each
tokens_llama_str = tokenizer.convert_ids_to_tokens(tokens_llama)  # Llama token strings
tokens_phi4_str = tokenizer_phi4.convert_ids_to_tokens(tokens_phi4)  # Phi-4 token strings
tokens_deepseek_str = tokenizer_deepseek.convert_ids_to_tokens(tokens_deepseek)  # Deepseek token strings
tokens_qwen_str = tokenizer_qwen.convert_ids_to_tokens(tokens_qwen)  # Qwen token strings

# Print side-by-side comparison table
print(f"\nTokenizing: '{text}'")
print("\n" + "=" * 140)
print(f"{'Pos':<5} | {'Llama 3.1':<25} | {'Phi-4':<25} | {'Deepseek':<25} | {'Qwen':<25}")
print(f"{'':<5} | {'ID':<12} {'Token':<12} | {'ID':<12} {'Token':<12} | {'ID':<12} {'Token':<12} | {'ID':<12} {'Token':<12}")
print("-" * 140)

# Find maximum length for alignment
max_len = max(len(tokens_llama), len(tokens_phi4), len(tokens_deepseek), len(tokens_qwen))

# Print aligned tokens side-by-side
for i in range(max_len):
    # Get token info for each model (or empty if shorter)
    llama_id = tokens_llama[i] if i < len(tokens_llama) else ""
    llama_token = tokens_llama_str[i] if i < len(tokens_llama_str) else ""
    
    phi4_id = tokens_phi4[i] if i < len(tokens_phi4) else ""
    phi4_token = tokens_phi4_str[i] if i < len(tokens_phi4_str) else ""
    
    deepseek_id = tokens_deepseek[i] if i < len(tokens_deepseek) else ""
    deepseek_token = tokens_deepseek_str[i] if i < len(tokens_deepseek_str) else ""
    
    qwen_id = tokens_qwen[i] if i < len(tokens_qwen) else ""
    qwen_token = tokens_qwen_str[i] if i < len(tokens_qwen_str) else ""
    
    # Print row with all models side-by-side
    print(f"{i:<5} | {str(llama_id):<12} {llama_token:<12} | {str(phi4_id):<12} {phi4_token:<12} | {str(deepseek_id):<12} {deepseek_token:<12} | {str(qwen_id):<12} {qwen_token:<12}")

print("=" * 140)

# Print summary statistics
print(f"\nSummary:")
print(f"  Llama 3.1:  {len(tokens_llama)} tokens")
print(f"  Phi-4:      {len(tokens_phi4)} tokens")
print(f"  Deepseek:   {len(tokens_deepseek)} tokens")
print(f"  Qwen:       {len(tokens_qwen)} tokens")