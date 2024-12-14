"""
Verify that models constructed with the positive embeddings mixin actually have positive embeddings.
"""
# built-in imports
import os

# libraries imports
import torch

# local imports
from utils.datasets_utils import load_split_dataset
from utils.general_utils import get_args
from utils.models_configs import model_name_from_config
from utils.models_inference import get_all_embeddings


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    # get args
    dataset_config, model_config, _ = get_args(positive=True)

    assert model_config.positive_embeddings, "Model should have positive embeddings for the test to be pertinent."

    # create model folders if not present
    models_folder = os.path.join(dataset_config.path, "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, model_name_from_config(model_config))
    os.makedirs(model_path, exist_ok=True)

    # check if a model already exists at model_path
    if not os.listdir(model_path):
        raise ValueError(f"\nNo model found at {model_path}, the model should have positive embeddings.")

    # load dataset
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_split_dataset(dataset_config)

    # load tokenizer
    tokenizer = model_config.tokenizer_class.from_pretrained(model_config.base_name, truncation=True, do_lower_case=True)

    # load the model
    model = model_config.model_class.from_pretrained(model_path, local_files_only=True)

    # load embeddings on cpu
    train_embeddings, val_embeddings, test_embeddings = get_all_embeddings(
        dataset_path=dataset_config.path, model_config=model_config, regenerate=False,
        model=model, tokenizer=tokenizer, device=DEVICE,
        train_inputs=train_data, val_inputs=val_data, test_inputs=test_data)
    
    # check if embeddings are positive
    assert torch.all(train_embeddings >= 0), "Train embeddings are not positive."
    assert torch.all(val_embeddings >= 0), "Val embeddings are not positive."
    assert torch.all(test_embeddings >= 0), "Test embeddings are not positive."


if __name__ == "__main__":
    main()
