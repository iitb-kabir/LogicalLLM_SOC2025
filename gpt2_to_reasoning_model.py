#  Import necessary modules
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

#  Load GPT-2 model and tokenizer with 4-bit quantization
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token for GPT-2

compute_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)

#  Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

#  Load and prepare the dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name)

# Rename 'text' to 'prompt' to match GRPOTrainer expectations
def rename_text_to_prompt(example):
    example['prompt'] = example['text']
    return example

train_dataset = dataset['train'].map(rename_text_to_prompt)

# Reward functions 
def reward_len(prompts, completions, completion_ids, **kwargs):
    scores = [len(comp) / 100.0 for comp in completions]
    return torch.tensor(scores, dtype=torch.float32)

def reward_token_diversity(prompts, completions, completion_ids, **kwargs):
    scores = [len(set(comp.split())) / 50.0 for comp in completions]
    return torch.tensor(scores, dtype=torch.float32)

def reward_reasoning(prompts, completions, completion_ids, **kwargs):
    keywords = ["because", "therefore", "thus", "hence", "consequently", "as a result"]
    scores = []
    for comp in completions:
        score = sum(1 for word in keywords if word in comp.lower())
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)

#  Configure GRPO training
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=512,
    max_completion_length=128,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# Initialize GRPO Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_len, reward_token_diversity, reward_reasoning],
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
output_dir = "./gpt2_reasoning_model"
model.save_pretrained(output_dir)  # Saves the LoRA adapters
tokenizer.save_pretrained(output_dir)  # Saves the tokenizer

#  Evaluate the model with reasoning prompts
eval_prompts = [
    "What is the next number in the sequence: 2, 4, 6, 8, ...?",
]

model.eval()
for prompt in eval_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=150)
    print(f"Prompt: {prompt}")
    print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")