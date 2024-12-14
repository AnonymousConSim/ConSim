import argparse
import pynvml

from .datasets_utils import get_dataset_config
from .models_configs import get_model_config


def get_free_memory_per_gpu():
    """Get free memory for each available GPU."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_memory = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory.append(mem_info.free)

    pynvml.nvmlShutdown()
    return free_memory


def select_best_gpu():
    """Select the GPU with the most free memory."""
    free_memory = get_free_memory_per_gpu()
    best_gpu = free_memory.index(max(free_memory))
    return best_gpu


def get_args(positive = None):
    """
    Function to parse terminal parameters
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument("--dataset", type=str, default="BIOS", help="Name of the dataset to use.")
    parser.add_argument("--model", type=str, default="distilbert", help="Name of the model to use.")
    parser.add_argument("--force", action="store_true", help="Force retraining of the model.")
    parser.add_argument("--positive", action="store_true", help="If embeddings should be positive.")
    parser.add_argument("--granularity", type=str, default="whole", help="Division of samples for embeddings.")
    parser.add_argument("--gradxinput", action="store_true", help="If concepts importances should be computed using gradxinput and not just grad.")

    args = parser.parse_args()

    if positive is not None:
        args.positive = positive

    assert args.granularity in ["whole", "sentence", "sentence-part", "unique-words"],\
        "Granularity must be one of 'whole', 'sentence', 'sentence-part', 'unique-words'."

    dataset_config = get_dataset_config(args.dataset)
    if args.model == "llama":
        assert not args.positive, "llama does not support specifying positive embeddings."
    
    model_config = get_model_config(args.model, args.dataset, positive=args.positive)

    return dataset_config, model_config, args