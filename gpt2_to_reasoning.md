# GRPO Fine-Tuning on GPT-2 with 4-bit Quantization and LoRA

This project demonstrates how to fine-tune a quantized GPT-2 language model using [GRPO (Group Relative Policy Optimization)](https://github.com/huggingface/trl) with reward functions targeting length, diversity, and reasoning capabilities.

---

##  Setup

###  Installation

Create a virtual environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate grpo-finetune
