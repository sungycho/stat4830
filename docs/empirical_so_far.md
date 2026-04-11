# Empirical Studies

## Results since Week 11
* During the presentation on April 7th, we discussed population-scale experiments, baseline differences for base models with and without in-context learning (prompts that provide base models with a few example dataset sequences to improve predictions), possible model collapse/reward hacking with binary rewards, and neural thickets.
* Since then, we plotted population-scale experiments for OPT models (350M, 1.3B, 13B) with base model friendly in-context learning prompting, and Llama-3.2 models (1B and 3B) as well as for Qwen-2.5-Instruct 1.5B on tasks like BoolQ, WSC, and DROP.
    * OPT and Qwen plots were inconclusive. No real learning seemed to be happening.
    * Despite being base models, the Llama models were run without in-context learning prompting. This happened because the Llama-3.2 models contain instruct templates embedded in the model itself, and our code does not perform base model prompting if this chat template is detected. We observed contradictory results where BoolQ actually performed better without it (with instruct model prompting instead), but WSC performed worse. We speculate that the latest Llama models are trained on more recent data that resembles the BoolQ dataset.
    * Llama 1B vs. 3B BoolQ plots offered some possible insight into a potential model size and population size relationship, with winner at N=16 for 1B and N=32 for 3B. However, subsequent follow-up with the WSC task proved inconclusive.
* Comparing our initial baseline of OPT on BoolQ (~0.1) to the MeZO paper's (~0.6), we discovered that most classification, multiple-choice, and question-and-answer tasks (not math or reasoning) use cross-entropy loss instead of binary rewards based on answers, as we had been doing.
* We are now trying to replicate the results and implementations in the [MeZO](https://arxiv.org/pdf/2305.17333) and [ES at Scale](https://arxiv.org/pdf/2509.24372) papers for our new hypothesis.

## Limitations
1. As we have to run experiments where the model sizes are big and the iterations are higher, our experiments are taking a lot of wall-clock time.
2. Some experiments, like the ES at Scale paper experiment with Qwen-2.5-Instruct on Countdown, are difficult to replicate as their setup includes multiple-GPU parallelization.
3. Something is wrong with our current MeZO implementation as seen through the discrepancy in our SST-2 runs on OPT-13B. We initially tried to integrate MeZO logic in our repo through augmentation, but we might have to call it separately.

## Next Steps
1. We will run a full (20,000 iterations/640,000 forward passes) experiment on OPT-13B and one or two tasks to explore our hypothesis about population indifference in MeZO for cross entropy tasks.
2. We will run population-scale experiments on Qwen-2.5-Instruct models to explore our hypothesis about the need for different population sizes for binary reward tasks.