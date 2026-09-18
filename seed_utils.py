import os
import random
import numpy as np
import torch


def set_seed(seed: int = 202601) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_utils] Đã fix seed = {seed} (deterministic mode)")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 202601) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g