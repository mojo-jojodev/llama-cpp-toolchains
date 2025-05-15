#!/bin/bash
# Automates the compilation of llama.cpp with CUDA support and GGUF conversion
set -e

echo "Compiling llama.cpp with CUBLAS (Nvidia Acceleration)..."
make LLAMA_CUBLAS=1 -j$(nproc)

echo "Converting Safetensors to GGUF format (FP16)..."
python3 convert_hf_to_gguf.py models/Mistral-7B-v0.1 --outfile models/Mistral-7B-v0.1-F16.gguf

echo "Quantizing to Q4_K_M for edge deployment..."
./quantize models/Mistral-7B-v0.1-F16.gguf models/Mistral-7B-v0.1-Q4_K_M.gguf Q4_K_M
