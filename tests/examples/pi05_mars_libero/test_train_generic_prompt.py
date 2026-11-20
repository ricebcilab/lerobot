import pytest
import torch
from train_generic_prompt import GENERIC_PROMPT, with_prompt_dropout


class Demos:
    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        return {"task": f"task {idx % 3}", "index": idx}


def test_prompt_dropout_swaps_the_task_sentence_for_the_generic_prompt_at_the_given_rate():
    torch.manual_seed(0)
    dataset = with_prompt_dropout(Demos(), 0.5)
    items = [dataset[i] for i in range(len(dataset))]
    generic = [item["task"] == GENERIC_PROMPT for item in items]
    assert 0.45 < sum(generic) / len(items) < 0.55
    assert all(
        g or item["task"] == f"task {item['index'] % 3}" for g, item in zip(generic, items, strict=True)
    )
    assert len(dataset) == 2000  # still the dataset it was

    assert all(with_prompt_dropout(Demos(), 0.0)[i]["task"] == f"task {i % 3}" for i in range(30))
    assert all(with_prompt_dropout(Demos(), 1.0, "anything")[i]["task"] == "anything" for i in range(30))
    with pytest.raises(ValueError):
        with_prompt_dropout(Demos(), 1.5)
