"""
Config and function for models
"""

from collections import namedtuple
import os
from typing import Tuple

from huggingface_hub import login
import torch
from torch import nn
import transformers

# from peft import LoraConfig, get_peft_model
# from transformers import BitsAndBytesConfig

from utils import splitted_models
from utils.datasets_utils import DatasetConfig

# define a namedtuple for configurations
ModelConfig = namedtuple(
    "Config",
    ["model_class", "tokenizer_class", "splitted_class", "base_name", "positive_embeddings", 
     "learning_rate", "num_train_epochs", "gradient_accumulation_steps", "max_length", "batch_size", "base_save_path"],
    defaults=(None,) * 11  # default values for all fields are None
)


# configurations for each model-dataset pair
__configs = {
    "default": {
        "default": ModelConfig(
            learning_rate=1e-05,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            max_length=128,
            batch_size=32,
        ),
        # "BIOS": ModelConfig(),
        "BIOS10": ModelConfig(),
        "IMDB": ModelConfig(),
        "tweet_eval_emotion": ModelConfig(num_train_epochs=5),
        "rotten_tomatoes": ModelConfig(num_train_epochs=3),
        "cola": ModelConfig(num_train_epochs=3, batch_size=64),
    },
    # "bert": {
    #     "default": ModelConfig(
    #         model_class=transformers.BertForSequenceClassification,
    #         tokenizer_class=transformers.BertTokenizer,
    #         splitted_class=splitted_models.SplittedBert,
    #         base_name="bert-base-uncased",
    #         learning_rate=2e-05,
    #         num_train_epochs=3,
    #         max_length=128,
    #         batch_size=16
    #     ),
    #     "BIOS": ModelConfig(),
    #     "IMDB": ModelConfig(
    #         batch_size=8,
    #     ),
    # },
    # "deberta": {
    #     "default": ModelConfig(
    #         model_class=transformers.DebertaV2ForSequenceClassification,
    #         tokenizer_class=transformers.DebertaV2Tokenizer,
    #         splitted_class=splitted_models.SplittedDeberta,
    #         base_name="microsoft/deberta-v3-base",
    #         learning_rate=2e-03,
    #         num_train_epochs=3,
    #         max_length=128,
    #         batch_size=16
    #     ),
    #     "BIOS": ModelConfig(),
    #     "IMDB": ModelConfig(),
    # },
    "distilbert": {
        "default": ModelConfig(
            model_class=transformers.DistilBertForSequenceClassification,
            tokenizer_class=transformers.DistilBertTokenizer,
            splitted_class=splitted_models.SplittedDistilBert,
            base_name="distilbert-base-uncased",
        ),
        "IMDB": ModelConfig(batch_size=16,),
        "tweet_eval_emotion": ModelConfig(batch_size=16),
    },
    "llama": {
        "default": ModelConfig(
            model_class=None,
            tokenizer_class=transformers.AutoTokenizer,
            splitted_class=splitted_models.SplittedLlamaForCausalLM,
            base_name="meta-llama/Meta-Llama-3-8B-Instruct",
            learning_rate=None,
            gradient_accumulation_steps=None,
            batch_size=None,
            base_save_path=None
        ),
    },
    # "llama-3-8B": {
    #     "default": ModelConfig(
    #         model_class=splitted_models.SplittedLlama,  # TODO: set back to:transformers.LlamaForSequenceClassification,
    #         tokenizer_class=transformers.AutoTokenizer,
    #         splitted_class=splitted_models.SplittedLlama,
    #         base_name="meta-llama/Meta-Llama-3-8B",
    #         learning_rate=1e-4,  # 1e-5
    #         gradient_accumulation_steps=4,
    #         batch_size=8,
    #         base_save_path=os.path.join(os.getcwd(), "data/Llama")
    #     ),
    #     "tweet_eval_emotion": ModelConfig(
    #         num_train_epochs=4,
    #         learning_rate=2e-4,
    #     ),
    #     "rotten_tomatoes": ModelConfig(
    #         learning_rate=2e-5
    #     ),
    # },
    # "roberta": {
    #     "default": ModelConfig(
    #         model_class=transformers.RobertaForSequenceClassification,
    #         tokenizer_class=transformers.RobertaTokenizer,
    #         splitted_class=splitted_models.SplittedRoberta,
    #         base_name="roberta-base",
    #         max_length=256,
    #         batch_size=16
    #     ),
    #     "BIOS": ModelConfig(),
    #     "IMDB": ModelConfig(),
    # },
    "t5": {
        "default": ModelConfig(
            model_class=transformers.T5ForSequenceClassification,
            tokenizer_class=transformers.T5Tokenizer,
            splitted_class=splitted_models.SplittedT5,
            base_name="t5-base",
            learning_rate=3e-4,
            batch_size=8
        ),
        "BIOS10": ModelConfig(batch_size=8),
        "BIOS": ModelConfig(batch_size=64),
        "BIOS_ng": ModelConfig(batch_size=64),
    },
}


def get_model_config(model_name: str, dataset_name: str, positive: bool = False) -> ModelConfig:
    """
    Get the config for the model and dataset using namedtuple.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    dataset_name : str
        Name of the dataset to use.
    positive : bool
        Whether to force positive embeddings or not.

    Returns
    -------
    Config
        Config for the model and dataset as a namedtuple.
    """
    assert model_name in __configs, f"Model {model_name} not in configs."
    # if dataset_name not in __configs[model_name]:
    #     print(f"Warning: dataset not in configs, using default config for {model_name}.")

    default_config = __configs["default"]["default"]
    default_dataset_config = __configs["default"].get(dataset_name, ModelConfig())
    model_default_config = __configs[model_name]["default"]
    dataset_specific_config = __configs[model_name].get(dataset_name, ModelConfig())

    # Create a combined Config namedtuple with overridden values
    combined_config = default_config._replace(**{k: v for k, v in default_dataset_config._asdict().items() if v is not None})
    combined_config = combined_config._replace(**{k: v for k, v in model_default_config._asdict().items() if v is not None})
    combined_config = combined_config._replace(**{k: v for k, v in dataset_specific_config._asdict().items() if v is not None})

    if positive:
        if model_name == "distilbert":
            new_class = splitted_models.PositiveDistilBert  # TODO, put this list in the configs
        elif model_name == "t5":
            new_class = splitted_models.PositiveT5
        else:
            raise NotImplementedError(f"Positive embeddings not implemented for model {model_name}.")
            # new_class = type(f"Positive{combined_config.model_class.__name__}",
            #                  (splitted_models.PositiveEmbeddingMixin, combined_config.splitted_class), {})
        combined_config = combined_config._replace(model_class=new_class, positive_embeddings=True)

    return combined_config


def create_model(model_config: ModelConfig, num_labels: int, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Create the model and tokenizer from the configuration.

    Parameters
    ----------
    model_config : ModelConfig
        Config for the model and dataset as a namedtuple.
    num_labels : int
        Number of labels for the classification task.
    device : torch.device
        Device to use for the model.

    Returns
    -------
    model : nn.Module
        The model.
    tokenizer : nn.Module
        The corresponding tokenizer.
    """
    # if "llama" in model_config.base_name.lower():
    #     return create_quantized_model(model_config, num_labels, device)
    model = model_config.model_class.from_pretrained(model_config.base_name, num_labels=num_labels).to(device)
    tokenizer = model_config.tokenizer_class.from_pretrained(model_config.base_name, truncation=True, do_lower_case=True)

    model.eval()
    return model, tokenizer


def get_splitted_model(config: ModelConfig,
                       dataset_config: DatasetConfig,
                       include_base: bool = True,
                       device: torch.device = torch.device("cpu")
                       ) -> Tuple[nn.Module, nn.Module]:
    """
    Get the splitted model from the configuration.

    Parameters
    ----------
    config : ModelConfig
        Config for the model and dataset as a namedtuple.
    dataset_config : str
        Path to the dataset.
    device : torch.device
        Device to use for the model.

    Returns
    -------
    model : nn.Module
        The splitted model.
    tokenizer : nn.Module
        The corresponding tokenizer.
    """
    
    if "llama" in config.base_name.lower():
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        # read token in hf_token.txt file
        token = open("hf_token.txt", "r").read().strip()
        login(token)

        # Suppress the warnings from the transformers library
        transformers.logging.set_verbosity_error()

        # download model from Hugging Face Hub
        model = config.splitted_class.from_pretrained(
            config.base_name,
            cache_dir=...,
            torch_dtype=torch.bfloat16,
            device_map=device,)
        tokenizer = config.tokenizer_class.from_pretrained(model_id, padding_side="left")

        model.setup(tokenizer, dataset_config)

        if not include_base:
            model.del_base_model()

        return model, tokenizer
    
    model_path = os.path.join(dataset_config.path, "models", model_name_from_config(config))
    if config.positive_embeddings:
        model_class = config.model_class
    else:
        model_class = config.splitted_class
    
    model = model_class.from_pretrained(model_path, num_labels=dataset_config.num_labels)
    tokenizer = config.tokenizer_class.from_pretrained(config.base_name)

    if not include_base:
        model.del_base_model()

    model.eval()
    return model, tokenizer


def model_name_from_config(config: ModelConfig) -> str:
    """
    Generate a name based on the configuration.
    This name takes into account the hyperparameters of the model.

    Parameters
    ----------
    config : Config
        Config for the model and dataset as a namedtuple.

    Returns
    -------
    str
        Name of the model for the save path.
    """
    model_name = config.base_name

    if "llama" in model_name.lower():
        model_name = "llama"
    else:
        
        model_name += f"_lr{config.learning_rate}"
        model_name += f"_epochs{config.num_train_epochs}"
        model_name += f"_maxlen{config.max_length}"
        model_name += f"_batch{config.batch_size}"

        if config.positive_embeddings:
            model_name += "_positive"

        # replace `/` by `_` in model name
        model_name = model_name.replace("/", "_")

    return model_name
