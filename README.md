# PokerGPT README.md generator
# PokerGPT: End-to-End Poker Solver Using LLMs

This project builds an end-to-end poker-playing agent using Large Language Models (LLMs) and Reinforcement Learning techniques. Inspired by [PokerGPT](https://arxiv.org/abs/2401.06781#:~:text=In%20this%20work%2C%20we%20introduce,large%20language%20model%20(LLM)), it focuses on multi-player Texas Hold’em, using GRPO (Generalized Reward Policy Optimization) or similar RL fine-tuning approaches to align the model’s play style with strategic objectives.

## 📁 Repository Structure

```
Poker_gpt(1).ipynb       # Main training and evaluation notebook
README.md             
```

## 🚀 Features

-  **Environment**: Simulated poker environment with betting, folding, and hand resolution.
-  **LLM-based Agent**: Uses pretrained instruction-tuned LLMs (gemma3-(2b)) as poker-playing agents.
-  **GRPO Training**: Fine-tunes the LLM using GRPO (Generalized Reward Policy Optimization).
-  **Evaluation**: Plots rewards, KL-divergence, and training progress.

## 🛠️ Setup

### 1. Install dependencies

Make sure you're using Google Colab or a machine with a GPU.

```bash
pip install unsloth trl datasets peft transformers accelerate einops
```

### 2. Model Requirements

Download or use a compatible HuggingFace model such as:

- `gemma3-(2b)`

### 3. Dataset

The training dataset should contain poker scenarios and optimal actions. A sample format:

```json
{
  "prompt": "Player 1 has AK, Player 2 bets 100, what should Player 1 do?",
  "completion": "raise"
}
```

## 📊 Outputs

- **Training logs**: Rewards, losses, and KL penalties per step.
- **Plots**: Automatically generated and saved in the `graphs/` directory.

## 📂 Save Graphs

Graphs are saved in a newly created `graphs` directory (created during runtime).

```python
import os
os.makedirs("graphs", exist_ok=True)
```

Example saving code:

```python
plt.savefig("graphs/reward_curve.png")
```

## 📌 To-Do

- [ ] Implement tournament-style evaluation.
- [ ] Improve RLHF reward modeling for bluffs and folds.
- [ ] Add curriculum learning for different poker variants.



## ✍️ Acknowledgements

- [PokerGPT PAPER](https://arxiv.org/abs/2401.06781#:~:text=In%20this%20work%2C%20we%20introduce,large%20language%20model%20(LLM))
- [LoRA PAPER](https://arxiv.org/abs/2106.09685)
- [QLoRA PAPER](https://arxiv.org/abs/2305.14314)
- [DeepSeekMath PAPER](https://arxiv.org/abs/2402.03300)
- [PPO Algorithms PAPER](https://arxiv.org/abs/1707.06347)
- HuggingFace Transformers & TRL Libraries
- Unsloth Team for efficient fine-tuning framework
"""

