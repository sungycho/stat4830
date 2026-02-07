import re

import verifiers as vf
from verifiers.envs.integrations.textarena_env import TextArenaEnv


DEFAULT_SYSTEM_PROMPT = """You are a competitive game player.
Read the game instructions carefully and follow the required format.

IMPORTANT:
- Output exactly ONE guess per turn.
- The guess MUST be wrapped in square brackets, like [crane].
- Do NOT include any other text.
"""


### feedback functions
def wordle_feedback_fn(observation: str) -> str:
    latest_observation = observation.split("[GAME]")[-1].strip()
    if "Feedback:" in latest_observation:
        return latest_observation.split("Feedback:")[-1]
    else:
        return latest_observation


### custom bracket parser
class BracketParser:
    def get_assistant_messages(self, completion):
        return [m for m in completion if m.get("role") == "assistant"]

    def get_user_messages(self, completion):
        return [m for m in completion if m.get("role") == "user"]

    def parse_answer(self, completion):
        """
        Extract the most recent [abcde] guess from assistant messages.
        """
        for m in self.get_assistant_messages(completion)[::-1]:
            text = m.get("content", "")
            match = re.search(r"\[([a-zA-Z]{5})\]", text)
            if match:
                return "[" + match.group(1).lower() + "]"
        return ""

    def get_format_reward_func(self):
        """
        Reward = 1 if the latest guess strictly matches [abcde].
        """
        def format_reward(parser, completion, answer, **kwargs):
            guess = parser.parse_answer(completion)
            return 1.0 if re.fullmatch(r"\[[a-z]{5}\]", guess) else 0.0

        format_reward.__name__ = "format_reward"
        return format_reward


### reward functions
def correct_answer(parser, completion, answer, **kwargs) -> float:
    """Whether the guess is exactly correct."""
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer + "]" else 0.0


def length_bonus(parser, completion, answer, **kwargs) -> float:
    """Bonus for shorter correct solutions."""
    assistant_messages = parser.get_assistant_messages(completion)
    guesses = [
        x for x in assistant_messages
        if re.search(r"\[[a-zA-Z]{5}\]", x.get("content", ""))
    ]
    is_correct = correct_answer(parser, completion, answer, **kwargs)
    return is_correct / (len(guesses) or 1)


def partial_answer(parser, completion, answer, **kwargs) -> float:
    """Partial credit based on the most recent feedback."""
    if correct_answer(parser, completion, answer, **kwargs):
        return 0.0

    user_messages = parser.get_user_messages(completion)
    for user_message in user_messages[::-1]:
        feedback = user_message.get("content", "").strip()
        feedback_parts = feedback.split("\n")
        if len(feedback_parts) == 3:
            _, scoring, _ = feedback_parts
            scoring = scoring.strip()
            num_greens = scoring.count("G")
            num_yellows = scoring.count("Y")
            return 0.2 * num_greens + 0.1 * num_yellows

    return 0.0


### environment loader
def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: int = 0,
    **kwargs,
):
    parser = BracketParser()

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(correct_answer)
    rubric.add_reward_func(partial_answer)
    rubric.add_reward_func(length_bonus)

    format_reward = parser.get_format_reward_func()
    rubric.add_reward_func(format_reward, weight=0.2)

    return TextArenaEnv(
        game="Wordle-v0",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        feedback_fn=wordle_feedback_fn,
        seed=seed,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
