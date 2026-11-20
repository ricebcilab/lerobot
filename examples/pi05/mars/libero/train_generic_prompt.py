"""Fine-tune the LIBERO policy with prompt dropout, so it is undecided under the generic prompt.

The released checkpoint always saw a task sentence. Under an unfamiliar prompt such as "do something" it falls
back on one fixed behaviour per scene, so steering has to force it off that default instead of choosing among
what it already offers. Prompt dropout is the remedy the cursor policy has by goal dropout: every training
sample keeps its task sentence with probability 1 - p and gets the generic prompt otherwise. Under the generic
prompt the policy then learns the mix of tasks the demonstrations show for that image (ten tasks in LIBERO-Goal's
one scene; a single task where a suite's layout gives the task away), and it still follows a task sentence.

This is `lerobot-train` with that one change, made where the dataset hands out the sentence; the policy, its
preprocessor and the saved checkpoint are the stock trainer's, so a session loads the result as any checkpoint:

    uv run --locked --extra pi --extra libero --extra training \
        python examples/pi05/mars/libero/train_generic_prompt.py --prompt-dropout 0.5 \
        --policy.path=lerobot/pi05_libero_finetuned --policy.push_to_hub=false \
        --dataset.repo_id=HuggingFaceVLA/libero --batch_size=32 --steps=6000 --save_freq=1000 \
        --output_dir=outputs/pi05_libero_generic_prompt

Every flag but `--prompt-dropout` and `--generic-prompt` is `lerobot-train`'s. For several GPUs, launch it with
`accelerate launch --num_processes=N`.
"""

import argparse
import sys

import torch

GENERIC_PROMPT = "do something"


def with_prompt_dropout(dataset, p: float, prompt: str = GENERIC_PROMPT):
    """`dataset` with each item's task sentence replaced by `prompt` with probability `p` (drawn from torch's
    generator, which every loader worker seeds on its own). Everything else about the dataset is unchanged."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"prompt dropout must be a probability, got {p}")

    class PromptDropout(type(dataset)):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            if torch.rand(()) < p:
                item["task"] = prompt
            return item

    dataset.__class__ = PromptDropout
    return dataset


def main() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    own = argparse.ArgumentParser(add_help=False)
    own.add_argument(
        "--prompt-dropout", type=float, default=0.5, help="share of samples given the generic prompt"
    )
    own.add_argument("--generic-prompt", default=GENERIC_PROMPT)
    args, rest = own.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # the stock trainer parses what is left

    make_dataset = lerobot_train.make_dataset

    def make_dataset_with_dropout(cfg):
        print(
            f"prompt dropout: '{args.generic_prompt}' replaces the task sentence with probability {args.prompt_dropout}"
        )
        return with_prompt_dropout(make_dataset(cfg), args.prompt_dropout, args.generic_prompt)

    lerobot_train.make_dataset = make_dataset_with_dropout
    lerobot_train.main()


if __name__ == "__main__":
    main()
