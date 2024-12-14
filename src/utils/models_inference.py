"""
Functions to compute embeddings of the inputs using a model.
"""

import json
import os
from tqdm import tqdm
from typing import Optional, Dict, List

import pandas as pd
import torch
from torch import nn

from utils.splitted_models import SplittedLlamaForCausalLM
from utils.models_configs import ModelConfig, model_name_from_config
from utils.text_utils import split_paragraphs, count_unique_words


def get_concepts_examples_embeddings(
        save_path: str,
        model_name: str,
        regenerate: bool = False,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[nn.Module] = None,
        max_length: int = 512,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        ) -> Dict[str, torch.Tensor]:
    """
    Compute the embeddings of the inputs using the model if they do not already exist.

    Parameters
    ----------
    save_path: str
        The path to save the embeddings.
    model_name: str
        The name of the model to use to compute the embeddings. Only for saving path purpose.
    regenerate: bool
        Whether to regenerate the embeddings or not.
    model: nn.Module
        The model to use to compute the embeddings.
    tokenizer: nn.Module
        The tokenizer to use to compute the embeddings.
    max_length: int
        The maximum length of the inputs.
    inputs: torch.Tensor
        The inputs to compute the embeddings.
    batch_size: int
        The batch size to use to compute the embeddings.
    device:
        The device to use to compute the embeddings.
    
    Returns
    -------
    embeddings: Dict[torch.Tensor]
        Dictionary of concepts input examples embeddings.
    """
    with open(os.path.join(save_path, "concepts_examples.json"), 'r') as json_file:
        concepts_examples = json.load(json_file)
    
    os.makedirs(os.path.join(save_path, "embeddings", model_name), exist_ok=True)

    concepts_embeddings = {}
    # iterate on the file concepts
    for concept_name, examples in concepts_examples.items():
        concept_name = concept_name.replace(" ", "_").lower()
        embeddings_save_path = os.path.join(save_path, "embeddings", model_name, f"{concept_name}_embeddings.pt")

        if os.path.exists(embeddings_save_path) and not regenerate:
            # load already computed embeddings for a given concept
            concepts_embeddings[concept_name] = torch.load(embeddings_save_path, map_location=torch.device("cpu")).to(torch.float32)
        else:
            model.to(device)
            if isinstance(model, SplittedLlamaForCausalLM):
                concept_embeddings = model.features(examples)
            else:
                batched_examples = [examples[i:i+batch_size] for i in range(0, len(examples), batch_size)]

                concept_embeddings = []
                with torch.no_grad():
                    # batch over sentences and compute features
                    for batch in batched_examples:
                        # tokenize the batch
                        tokenized_batch = tokenizer(batch, padding="max_length",
                                                    max_length=max_length, truncation=True,
                                                    return_tensors='pt')
                        tokenized_batch.to(device)

                        # compute the embeddings
                        batch_embeddings = model.features(**tokenized_batch).cpu()
                        concept_embeddings.append(batch_embeddings.to(torch.float32))

                # concatenate the batch embeddings
                concept_embeddings = torch.cat(concept_embeddings)

            # save the embeddings
            torch.save(concept_embeddings, embeddings_save_path)

            # add embeddings to the dictionary
            concepts_embeddings[concept_name] = concept_embeddings
    return concepts_embeddings


def get_unique_words_embeddings(dataset_path: str,
                                model_config: ModelConfig,
                                regenerate: bool = False,
                                model: Optional[nn.Module] = None,
                                tokenizer: Optional[nn.Module] = None,
                                max_length: int = 512,
                                unique_words: List[str] = None,
                                batch_size: Optional[int] = None,
                                device: Optional[torch.device] = None,
                                tqdm_disable: Optional[bool] = True,
                                ) -> torch.Tensor:
    """
    Compute the embeddings of the inputs using the model if they do not already exist.

    Parameters
    ----------
    dataset_path: str
        The path to the dataset. (Models are stored at the same place).
    model_config: ModelConfig
        The model configuration.
    regenerate: bool
        Whether to regenerate the embeddings or not.
    model: nn.Module
        The model to use to compute the embeddings.
    tokenizer: nn.Module
        The tokenizer to use to compute the embeddings.
    max_length: int
        The maximum length of the inputs.
    unique_words: List[str]
        The unique words to compute the embeddings.
    batch_size: int
        The batch size to use to compute the embeddings.
    device:
        The device to use to compute the embeddings.
    tqdm_disable: bool
        Whether to disable the tqdm progress bar.

    Returns
    -------
    embeddings: torch.Tensor
        The embeddings of the unique words.
    """
    # create path
    save_folder = os.path.join(dataset_path,
                               "models",
                               model_name_from_config(model_config),
                               "embeddings")
    os.makedirs(save_folder, exist_ok=True)
    embeddings_save_path = os.path.join(save_folder, "unique_words_embeddings.pt")

    if os.path.exists(embeddings_save_path) and not regenerate:
        # load already computed embeddings and test length coherence
        embeddings = torch.load(embeddings_save_path, map_location=torch.device("cpu")).to(torch.float32)
        if unique_words is not None:
            assert len(embeddings) == len(unique_words),\
                "The number of unique words should be the same as the number of embeddings."+\
                f"Got {len(embeddings)} embeddings and {len(unique_words)} unique words."
    else:
        model.to(device)
        if isinstance(model, SplittedLlamaForCausalLM):
            model.tqdm = not tqdm_disable
            embeddings = model.features(batch)
        else:
            batched_unique_words = [unique_words[i:i+batch_size] for i in range(0, len(unique_words), batch_size)]

            embeddings = []
            with torch.no_grad():
                # batch over sentences and compute features
                for batch in batched_unique_words:
                    # tokenize the batch
                    tokenized_batch = tokenizer(batch, padding="max_length",
                                                max_length=max_length, truncation=True,
                                                return_tensors='pt')
                    tokenized_batch.to(device)

                    # compute the embeddings
                    batch_embeddings = model.features(**tokenized_batch).cpu()
                    embeddings.append(batch_embeddings.to(torch.float32))

            # concatenate the batch embeddings
            embeddings = torch.cat(embeddings)

        # save the embeddings
        torch.save(embeddings, embeddings_save_path)
    return embeddings


def get_embeddings(save_path: str,
                   regenerate: bool = False,
                   model: Optional[nn.Module] = None,
                   tokenizer: Optional[nn.Module] = None,
                   max_length: int = 512,
                   inputs: Optional[torch.Tensor] = None,
                   batch_size: Optional[int] = None,
                   device: Optional[torch.device] = None,
                   granularity: str = "whole",
                   dataset_path: Optional[str] = None) -> torch.Tensor:
    """
    Compute the embeddings of the inputs using the model if they do not already exist.

    Parameters
    ----------
    save_path: str
        The path to save the embeddings.
    regenerate: bool
        Whether to regenerate the embeddings or not.
    model: nn.Module
        The model to use to compute the embeddings.
    tokenizer: nn.Module
        The tokenizer to use to compute the embeddings.
    max_length: int
        The maximum length of the inputs.
    inputs: torch.Tensor
        The inputs to compute the embeddings.
    batch_size: int
        The batch size to use to compute the embeddings.
    device:
        The device to use to compute the embeddings.
    granularity: str
        The granularity of the inputs to compute the embeddings.
    
    Returns
    -------
    embeddings: torch.Tensor
        The embeddings of the inputs.
    """
    if granularity == "whole":
        pass
    elif granularity in ["sentence", "sentence-part"]:
        save_path = save_path.replace(".pt", f"_{granularity}.pt")
    else:
        raise ValueError(f"granularity should be 'whole', 'sentence', or 'sentence-part', not '{granularity}'.")
    
    # print("DEBUG: get_embeddings: save_path", save_path)

    if regenerate or not os.path.exists(save_path):
        if inputs is None:
            return None
        model.eval()
        model.to(device)
        embeddings = []
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs["sentence"]
        
        if granularity in ["sentence", "sentence-part"]:
            inputs = split_paragraphs(inputs, granularity, save_dir=os.path.dirname(save_path))

        with torch.no_grad():
            # batch over sentences and compute features
            for i in tqdm(range(0, len(inputs), batch_size)):
                # last batch might not be of batch_size
                last = min(i + batch_size, len(inputs))
                batch_samples = [str(pd_input) for pd_input in inputs[i:last]]  # inputs["sentence"][i:last]

                # tokenize the batch
                tokenized_batch = tokenizer(batch_samples, padding="max_length",
                                            max_length=max_length, truncation=True,
                                            return_tensors='pt')
                tokenized_batch.to(device)

                # compute the embeddings
                batch_embeddings = model.features(**tokenized_batch).detach().cpu()
                embeddings.append(batch_embeddings.to(torch.float32))

        # concatenate the batch embeddings
        embeddings = torch.cat(embeddings)

        # save the embeddings
        torch.save(embeddings, save_path)
    else:
        embeddings = torch.load(save_path, map_location=torch.device("cpu"))

        # convert to float32
        embeddings = embeddings.to(torch.float32)
    return embeddings


def get_all_embeddings(dataset_path: str,
                       model_config: ModelConfig,
                       regenerate: bool = False,
                       model: Optional[nn.Module] = None,
                       tokenizer: Optional[nn.Module] = None,
                       train_inputs: Optional[torch.Tensor] = None,
                       val_inputs: Optional[torch.Tensor] = None,
                       test_inputs: Optional[torch.Tensor] = None,
                       device: Optional[torch.device] = None,
                       batch_size: Optional[int] = None,
                       granularity: str = "whole") -> torch.Tensor:
    """
    Compute the embeddings of the inputs using the model if they do not already exist.

    Parameters
    ----------
    dataset_path: str
        The path to the dataset. (Models are stored at the same place).
    model_config: ModelConfig
        The model configuration.
    regenerate: bool
        Whether to regenerate the embeddings or not.
    model: nn.Module
        The model to use to compute the embeddings.
    tokenizer: nn.Module
        The tokenizer to use to compute the embeddings.
    train_inputs: torch.Tensor
        The training inputs to compute the embeddings.
    val_inputs: torch.Tensor
        The validation inputs to compute the embeddings.
    test_inputs: torch.Tensor
        The test inputs to compute the embeddings.
    device:
        The device to use to compute the embeddings.
    batch_size: int
        The batch size to use to compute the embeddings.
    granularity: str
        The granularity of the inputs to compute the embeddings.
        If "whole", the embeddings are computed for the whole inputs.
        If "sentence", the embeddings are computed for each sentence of the inputs.
        If "sentence-part", the embeddings are computed for each part of the sentences of the inputs.
    
    Returns
    -------
    train_embeddings: torch.Tensor
        The embeddings of the training inputs.
    val_embeddings: torch.Tensor
        The embeddings of the validation inputs.
    test_embeddings: torch.Tensor
        The embeddings of the test inputs.
    """
    save_folder = os.path.join(dataset_path,
                               "models",
                               model_name_from_config(model_config),
                               "embeddings")
    os.makedirs(save_folder, exist_ok=True)

    if batch_size is None:
        batch_size = model_config.batch_size * 8  # TODO check if this is a good value

    train_embeddings = get_embeddings(os.path.join(save_folder, "train_embeddings.pt"), regenerate,
                                      model, tokenizer, model_config.max_length,
                                      train_inputs, batch_size, device, granularity, dataset_path)
    val_embeddings = get_embeddings(os.path.join(save_folder, "val_embeddings.pt"), regenerate,
                                    model, tokenizer, model_config.max_length,
                                    val_inputs, batch_size, device, granularity, dataset_path)
    test_embeddings = get_embeddings(os.path.join(save_folder, "test_embeddings.pt"), regenerate,
                                     model, tokenizer, model_config.max_length,
                                     test_inputs, batch_size, device, granularity, dataset_path)
    return train_embeddings, val_embeddings, test_embeddings


def get_logits_from_embeddings(model: nn.Module,
                               embeddings: torch.Tensor,
                               batch_size: int,
                               device: torch.device) -> torch.Tensor:
    """
    Compute the logits from the embeddings using the model.

    Parameters
    ----------
    model: nn.Module
        The model to use to compute the logits.
    embeddings: torch.Tensor
        The embeddings to compute the logits.
    device: torch.device
        The device to use to compute the logits.
    
    Returns
    -------
    logits: torch.Tensor
        The logits of the embeddings.
    """
    model.eval()
    model.end_model_to(device)
    model.end_model_to(torch.float32)
    with torch.no_grad():
        # batch over embeddings
        logits = []
        for i in tqdm(range(0, len(embeddings), batch_size)):
            # last batch might not be of batch_size
            last = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:last].to(device)
            batch_logits = model.end_model(batch_embeddings).detach().cpu()
            del batch_embeddings
            logits.append(batch_logits)
        logits = torch.cat(logits)
    return logits
