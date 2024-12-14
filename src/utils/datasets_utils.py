"""
Load data and create splits for the different datasets.
"""

from collections import namedtuple
import os
import pickle
from typing import Optional, Tuple

import pandas as pd
import torch
from torch import nn
from datasets import Dataset, Features, ClassLabel, Value
from datasets import load_dataset, load_from_disk

from fast_ml.model_development import train_valid_test_split

ROOT_DATASET_FOLDER = "/datasets/shared_datasets"

# define a namedtuple for configurations
DatasetConfig = namedtuple(
    "Config",
    ["name", "path", "files_names", "labels_names", "num_labels"],
    defaults=(None,) * 5  # default values for all fields are None
)

# configurations for each dataset
__configs = {
    "BIOS": DatasetConfig(name="BIOS",
                          path=os.path.join(ROOT_DATASET_FOLDER, "BIOS"),
                          files_names={"data": "data.pkl", "labels": "labels.pkl"},
                          labels_names=["surgeon", "pastor", "photographer", "professor", "chiropractor", "software_engineer",
                                        "teacher", "poet", "dj", "rapper", "paralegal", "physician", "journalist", "architect",
                                        "attorney", "yoga_teacher", "nurse", "painter", "model", "composer",
                                        "personal_trainer", "filmmaker", "comedian", "accountant", "interior_designer",
                                        "dentist", "psychologist", "dietitian"],),
    "BIOS_ng": DatasetConfig(name="BIOS_ng",
                             path=os.path.join(ROOT_DATASET_FOLDER, "BIOS_ng"),
                             files_names={"data": "data.pkl", "labels": "labels.pkl"},
                             labels_names=["surgeon", "pastor", "photographer", "professor", "chiropractor", "software_engineer",
                                           "teacher", "poet", "dj", "rapper", "paralegal", "physician", "journalist", "architect",
                                           "attorney", "yoga_teacher", "nurse", "painter", "model", "composer",
                                           "personal_trainer", "filmmaker", "comedian", "accountant", "interior_designer",
                                           "dentist", "psychologist", "dietitian"],),
    "BIOS10": DatasetConfig(name="BIOS10",
                           path=os.path.join(ROOT_DATASET_FOLDER, "BIOS10"),
                           files_names={"data": "data.pkl", "labels": "labels.pkl"},
                           labels_names=["surgeon", "photographer", "professor",
                                        "teacher", "physician", "journalist",
                                        "attorney", "nurse", "dentist", "psychologist"],),
    "IMDB": DatasetConfig(name="IMDB",
                          path=os.path.join(ROOT_DATASET_FOLDER, "IMDB"),
                          labels_names=["neg", "pos"],),
    "tweet_eval_emotion": DatasetConfig(name="tweet_eval_emotion",  # https://huggingface.co/datasets/tweet_eval
                                        path=os.path.join(ROOT_DATASET_FOLDER, "tweet_eval_emotion"),
                                        labels_names=["anger", "joy", "optimism", "sadness"],),
    "rotten_tomatoes": DatasetConfig(name="rotten_tomatoes",
                                     path=os.path.join(ROOT_DATASET_FOLDER, "rotten_tomatoes"),
                                     labels_names=["neg", "pos"],),
    "cola": DatasetConfig(name="cola",
                          path=os.path.join(ROOT_DATASET_FOLDER, "cola"),
                          labels_names=["unacceptable", "acceptable"],),
}

# update configurations with number of labels
for dataset, config in __configs.items():
    if config.labels_names is not None:
        __configs[dataset] = config._replace(**{"num_labels": len(config.labels_names)})


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get the configuration for the dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset.

    Returns
    -------
    config
        Configuration for the dataset.
    """

    if dataset_name not in __configs:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return __configs[dataset_name]


def load_hf_dataset(dataset_config: DatasetConfig) -> Tuple[Dataset, Dataset]:
    """
    Load the IMDB dataset.

    Parameters
    ----------
    dataset_config
        Configuration for the IMDB dataset.

    Returns
    -------
    datasets_tuple
        A tuple (train_dataset, test_dataset) of Hugging Face Datasets.
    """
    # load dataset
    dataset = load_from_disk(dataset_config.path)

    return dataset["train"], dataset["test"]


def load_bios_dataset(bios_config: DatasetConfig,
                      split: tuple = (0.7, 0.1, 0.2),
                      random_state: int = 0,
                      ) -> pd.DataFrame:
    """
    Load the BIOS dataset.

    Parameters
    ----------
    bios_config
        Configuration for the BIOS dataset.

    Returns
    -------
    splits
        A tuple (train_data, train_labels, val_data, val_labels, test_data, test_labels) of dataframes.
    """
    # load data and labels
    data = pickle.load(open(os.path.join(bios_config.path, "data.pkl"), "rb"))
    labels = pickle.load(open(os.path.join(bios_config.path, "labels.pkl"), "rb"))

    # conver data and labels to dataframe
    dataframe = pd.DataFrame(data, columns=["sentence"])
    labels_name2id = {name: i for i, name in enumerate(bios_config.labels_names)}
    dataframe["label"] = [labels_name2id[name] for name in labels]

    # split dataframe into train, validation and test
    splits = train_valid_test_split(dataframe, target="label", method="random",
                                    train_size=split[0], valid_size=split[1], test_size=split[2],
                                    random_state=random_state)

    return splits


def load_bios10_dataset(bios10_config: DatasetConfig,
                        split: tuple = (0.7, 0.1, 0.2),
                        random_state: int = 0,
                        ) -> pd.DataFrame:
    """
    Load the BIOS10 dataset, the 10 most frequent classes from the BIOS dataset.

    Parameters
    ----------
    bios10_config
        Configuration for the BIOS dataset.

    Returns
    -------
    splits
        A tuple (train_data, train_labels, val_data, val_labels, test_data, test_labels) of dataframes.
    """
    # load data and labels
    data = pickle.load(open(os.path.join(bios10_config.path[:-2], "data.pkl"), "rb"))
    labels = pickle.load(open(os.path.join(bios10_config.path[:-2], "labels.pkl"), "rb"))

    # mask for the 10 most frequent classes
    frequent_classes_mask = [(label in bios10_config.labels_names) for label in labels]
    data_subset = [data[i] for i, mask in enumerate(frequent_classes_mask) if mask]
    labels_subset = [labels[i] for i, mask in enumerate(frequent_classes_mask) if mask]

    del data
    del labels

    # convert data and labels subsets to dataframe
    dataframe = pd.DataFrame(data_subset, columns=["sentence"])
    labels_name2id = {name: i for i, name in enumerate(bios10_config.labels_names)}
    dataframe["label"] = [labels_name2id[name] for name in labels_subset]

    # split dataframe into train, validation and test
    splits = train_valid_test_split(dataframe, target="label", method="random",
                                    train_size=split[0], valid_size=split[1], test_size=split[2],
                                    random_state=random_state)

    return splits


def load_split_dataset(dataset_config: DatasetConfig) -> tuple:
    """
    Load the dataset and split it into train, validation and test sets.

    Parameters
    ----------
    dataset_config
        Configuration for the dataset.
    split
        A tuple of floats representing the split ratio for the train, validation and test sets.
    random_state
        Random state for reproducibility.

    Returns
    -------
    splits
        A tuple (train_data, train_labels, val_data, val_labels, test_data, test_labels) of dataframes.
    """
    if dataset_config.name in ["BIOS", "BIOS_ng"]:
        splits = load_bios_dataset(dataset_config)
    elif dataset_config.name == "BIOS10":
        splits = load_bios10_dataset(dataset_config)
    elif dataset_config.name in ["IMDB", "tweet_eval_emotion", "rotten_tomatoes", "cola"]:
        train_dataset, test_dataset = load_hf_dataset(dataset_config)
        splits = train_dataset['text'], train_dataset['label'], None, None, test_dataset['text'], test_dataset['label']
    else:
        raise ValueError(f"Dataset {dataset_config.name} not supported.")

    return splits


def load_huggingface_dataset_parts(dataset_config: DatasetConfig) -> tuple:
    """
    Load the dataset from Hugging Face.

    Parameters
    ----------
    dataset_config
        Configuration for the dataset.

    Returns
    -------
    splits
        A tuple (train_dataset, test_dataset) of Hugging Face Datasets.
    """
    if dataset_config.name in ["BIOS", "BIOS_ng"]:
        splits = load_bios_dataset(dataset_config)
        train_dataset = transform_dataset(splits[0], splits[1], None, dataset_config)
        test_dataset = transform_dataset(splits[2], splits[3], None, dataset_config)
    elif dataset_config.name == "BIOS10":
        splits = load_bios10_dataset(dataset_config)
        train_dataset = transform_dataset(splits[0], splits[1], None, dataset_config)
        test_dataset = transform_dataset(splits[2], splits[3], None, dataset_config)
    elif dataset_config.name in ["IMDB", "tweet_eval_emotion", "rotten_tomatoes", "cola"]:
        train_dataset, test_dataset = load_hf_dataset(dataset_config)
        if dataset_config.name == "cola":
            # rename sentence column to text and remove idx column
            train_dataset = train_dataset.rename_column("sentence", "text").remove_columns("idx")
            test_dataset = test_dataset.rename_column("sentence", "text").remove_columns("idx")
    else:
        raise ValueError(f"Dataset {dataset_config.name} not supported.")

    return train_dataset, test_dataset


def transform_dataset(data: pd.DataFrame,
                      labels: pd.Series,
                      tokenizer: Optional[nn.Module],
                      dataset_config: DatasetConfig,
                      max_length: int = 512,
                      ) -> Dataset:
    """
    Transform the data and labels into a Hugging Face Dataset.

    Parameters
    ----------
    data
        Data to be transformed.
    labels
        Labels to be transformed.
    tokenizer
        Tokenizer to be used to encode the data.
    dataset_config
        Configuration for the dataset.
    max_length
        Maximum length of the input sequence.

    Returns
    -------
    dataset
        Hugging Face Dataset with tokenized inputs.
    """
    # initialize dataset from data
    dataset = Dataset.from_pandas(data, preserve_index=False)

    # rename data column to "data"
    data_column_name = data.columns[0]
    dataset = dataset.rename_column(data_column_name, "data")

    # add labels column
    dataset = dataset.add_column("labels", labels).with_format("torch")

    # set features
    features = Features({"data": Value("string"),
                         "labels": ClassLabel(names=dataset_config.labels_names)})
    dataset.cast(features)

    # tokenize data and apply one hot encoding to labels
    if tokenizer is not None:
        dataset = dataset.map(lambda batch: {
            "labels": batch["labels"],
            **tokenizer(batch["data"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        }, batched=True)
    else:
        # for testing we do not encode data
        dataset = dataset.map(lambda batch: {
            "labels": batch["labels"],
            "text": batch["data"],
        }, batched=True)

    # remove data column as it was tokenized
    dataset = dataset.remove_columns(["data"])

    return dataset
