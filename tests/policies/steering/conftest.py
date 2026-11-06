"""Keep torch's threads few for the steering tests (their tensors are tiny: many threads only contend)."""

import os

import torch


def pytest_configure(config):
    torch.set_num_threads(int(os.environ.get("MARS_TEST_THREADS", min(16, os.cpu_count() or 16))))
