"""Main entry point for Wordle policy gradient training."""

from src.policy_gradient import PolicyGradientTrainer, TrainingConfig


def main():
    """Run policy gradient training on Wordle environment."""
    # Configuration
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B",  # Small model for testing
        learning_rate=1e-5,
        batch_size=4,
        num_epochs=5,  # Start with fewer epochs for testing
        num_train_examples=50,  # Start small
        num_eval_examples=10,
        seed=42,
    )
    
    # Create trainer
    trainer = PolicyGradientTrainer(config)
    
    # Train
    trainer.train()
    
    # Evaluate
    trainer.evaluate()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
