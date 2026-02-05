"""
Policy Gradient Trainer for Wordle using Qwen model.
Implements REINFORCE algorithm for training LLM policies.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

from .wordle import load_environment


@dataclass
class TrainingConfig:
    """Configuration for policy gradient training."""
    model_name: str = "Qwen/Qwen2.5-0.5B"  # Small Qwen model for testing
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 10
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_grad_norm: float = 1.0
    num_train_examples: int = 100  # Start small for testing
    num_eval_examples: int = 20
    seed: int = 42


class PolicyGradientTrainer:
    """REINFORCE policy gradient trainer for LLM policies."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type == "cpu":
            self.model.to(self.device)
        self.model.train()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        # Load environment
        self.env = load_environment(
            num_train_examples=config.num_train_examples,
            num_eval_examples=config.num_eval_examples,
            seed=config.seed,
        )
        
    def generate_completion(self, prompt: str, track_grad: bool = True) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a completion from the model given a prompt.
        Returns (generated_text, input_ids, log_probs_sum).
        """
        # Prepare input for the model
        messages = [
            {"role": "system", "content": self.env.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Format messages for tokenizer
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        
        # Generate tokens one by one to track log probs
        generated_ids = input_ids.clone()
        log_probs_list = []
        
        with torch.set_grad_enabled(track_grad):
            for _ in range(50):  # Max 50 new tokens
                # Forward pass
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :]  # Last token logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Sample next token
                probs = F.softmax(logits / 0.7, dim=-1)  # Temperature sampling
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Get log prob of sampled token
                log_prob = log_probs.gather(1, next_token)
                log_probs_list.append(log_prob)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Sum of log probabilities
        log_probs_sum = torch.stack(log_probs_list).sum() if log_probs_list else torch.tensor(0.0, device=self.device)
        
        return generated_text, input_ids, log_probs_sum
    
    def generate_trajectory(self, example_idx: int) -> Tuple[Dict, float]:
        """
        Generate a trajectory by interacting with the environment.
        Returns (trajectory_dict, total_reward).
        """
        # Get the example from the environment
        if hasattr(self.env, 'train_examples') and example_idx < len(self.env.train_examples):
            example = self.env.train_examples[example_idx]
            prompt = str(example) if not isinstance(example, dict) else example.get("prompt", str(example))
        else:
            prompt = f"Solve this Wordle puzzle. Example {example_idx}"
        
        # Generate completion
        generated_text, input_ids, log_probs_sum = self.generate_completion(prompt, track_grad=True)
        
        # Compute reward using environment's rubric
        # The verifiers Rubric aggregates reward functions
        try:
            # Get the example and answer
            if hasattr(self.env, 'train_examples') and example_idx < len(self.env.train_examples):
                example = self.env.train_examples[example_idx]
                if isinstance(example, dict):
                    answer = example.get("answer", "")
                else:
                    # Try to extract answer from example object
                    answer = getattr(example, 'answer', '')
            else:
                answer = ""
            
            parser = self.env.parser
            
            # Try to compute reward - the RubricGroup might have a different API
            # First, try to see if it has a method to compute rewards
            reward = 0.0
            
            # Method 1: Try calling the rubric directly (if it's callable)
            if callable(self.env.rubric):
                try:
                    reward = self.env.rubric(parser, generated_text, answer)
                except (TypeError, AttributeError) as e:
                    # If that fails, try other approaches
                    pass
            
            # Method 2: Try accessing reward functions directly from RubricGroup
            if reward == 0.0:
                # RubricGroup might store functions in different ways
                # Try to get the reward functions list
                reward_funcs = None
                
                # Check common attribute names
                for attr_name in ['_reward_funcs', 'reward_funcs', '_funcs', 'funcs', 'functions', 'reward_functions']:
                    if hasattr(self.env.rubric, attr_name):
                        attr_value = getattr(self.env.rubric, attr_name)
                        if attr_value is not None:
                            reward_funcs = attr_value
                            break
                
                # If we found reward functions, call them
                if reward_funcs:
                    for func_info in reward_funcs:
                        # Handle different formats: (func, weight) or just func
                        if isinstance(func_info, tuple) and len(func_info) >= 2:
                            func, weight = func_info[0], func_info[1]
                        elif isinstance(func_info, tuple) and len(func_info) == 1:
                            func, weight = func_info[0], 1.0
                        else:
                            func, weight = func_info, 1.0
                        
                        try:
                            # Call reward function with parser, completion, answer
                            # This matches the signature in wordle.py
                            func_reward = func(parser, generated_text, answer)
                            reward += func_reward * weight
                        except Exception as func_e:
                            # Skip if function fails
                            continue
                            
            # Method 3: Try using the parser's reward computation if available
            if reward == 0.0 and hasattr(parser, 'compute_reward'):
                try:
                    reward = parser.compute_reward(generated_text, answer)
                except:
                    pass
            
            # Method 4: Manually call the reward functions from wordle.py
            # Import them directly to ensure we can compute rewards
            if reward == 0.0:
                try:
                    from .wordle import correct_answer, partial_answer, length_bonus
                    
                    # Get format reward from parser
                    format_reward = parser.get_format_reward_func()
                    
                    # Compute each reward component
                    reward = (
                        correct_answer(parser, generated_text, answer) +
                        partial_answer(parser, generated_text, answer) +
                        length_bonus(parser, generated_text, answer) +
                        0.2 * format_reward(parser, generated_text, answer)  # weight from wordle.py
                    )
                except Exception as manual_e:
                    # If manual computation also fails, we'll keep reward as 0.0
                    pass
                            
        except Exception as e:
            # Print warning with debug info only once (use a flag to prevent spam)
            if not hasattr(self, '_reward_warning_shown'):
                rubric_type = self.env.rubric.__class__.__name__ if hasattr(self.env.rubric, '__class__') else 'Unknown'
                available_attrs = [attr for attr in dir(self.env.rubric) if not attr.startswith('__')][:15]
                print(f"Warning: Could not compute reward using rubric API: {e}")
                print(f"  Rubric type: {rubric_type}")
                print(f"  Trying manual reward computation...")
                print(f"  Available rubric attributes: {available_attrs}")
                self._reward_warning_shown = True
            
            # Try manual computation as fallback
            try:
                from .wordle import correct_answer, partial_answer, length_bonus
                parser = self.env.parser
                format_reward = parser.get_format_reward_func()
                reward = (
                    correct_answer(parser, generated_text, answer) +
                    partial_answer(parser, generated_text, answer) +
                    length_bonus(parser, generated_text, answer) +
                    0.2 * format_reward(parser, generated_text, answer)
                )
            except:
                reward = 0.0
        
        trajectory = {
            "input_ids": input_ids,
            "generated_text": generated_text,
            "log_probs_sum": log_probs_sum,
            "reward": reward,
        }
        
        return trajectory, reward
    
    def compute_policy_loss(self, trajectory: Dict) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss.
        Loss = -log π(a|s) * R
        """
        log_probs = trajectory["log_probs_sum"]
        reward = trajectory["reward"]
        
        # REINFORCE loss: -log π(a|s) * R
        loss = -log_probs * reward
        
        return loss
    
    def train_step(self, example_indices: List[int]) -> Dict[str, float]:
        """Perform one training step on a batch of examples."""
        self.model.train()
        total_loss = 0.0
        total_rewards = []
        
        for example_idx in example_indices:
            # Generate trajectory
            trajectory, reward = self.generate_trajectory(example_idx)
            total_rewards.append(reward)
            
            # Compute loss
            loss = self.compute_policy_loss(trajectory)
            total_loss += loss
        
        # Average loss
        avg_loss = total_loss / len(example_indices) if example_indices else torch.tensor(0.0)
        
        # Backward pass
        if avg_loss.item() != 0:
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": avg_loss.item(),
            "avg_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "std_reward": np.std(total_rewards) if total_rewards else 0.0,
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.config.num_train_examples} examples")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Create batches
            example_indices = list(range(self.config.num_train_examples))
            np.random.shuffle(example_indices)
            
            epoch_losses = []
            epoch_rewards = []
            
            # Process in batches
            for i in range(0, len(example_indices), self.config.batch_size):
                batch_indices = example_indices[i:i + self.config.batch_size]
                metrics = self.train_step(batch_indices)
                
                epoch_losses.append(metrics["loss"])
                epoch_rewards.append(metrics["avg_reward"])
                
                print(
                    f"  Batch {i // self.config.batch_size + 1}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Reward={metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}"
                )
            
            print(
                f"Epoch {epoch + 1} Summary: "
                f"Avg Loss={np.mean(epoch_losses):.4f}, "
                f"Avg Reward={np.mean(epoch_rewards):.4f}"
            )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the policy on the evaluation set."""
        self.model.eval()
        eval_rewards = []
        
        print("\nEvaluating on validation set...")
        with torch.no_grad():
            for example_idx in range(self.config.num_eval_examples):
                _, reward = self.generate_trajectory(example_idx)
                eval_rewards.append(reward)
                if (example_idx + 1) % 5 == 0:
                    print(f"  Evaluated {example_idx + 1}/{self.config.num_eval_examples}")
        
        metrics = {
            "eval_avg_reward": np.mean(eval_rewards),
            "eval_std_reward": np.std(eval_rewards),
            "eval_success_rate": np.mean([r > 0 for r in eval_rewards]),
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {metrics['eval_avg_reward']:.4f} ± {metrics['eval_std_reward']:.4f}")
        print(f"  Success Rate: {metrics['eval_success_rate']:.2%}")
        
        return metrics
