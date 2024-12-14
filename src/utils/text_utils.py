"""
Split paragraphs into smaller parts (sentences, words, clauses).
"""

from collections import Counter
import os
import json
from tqdm import tqdm
from typing import Dict, List, Union, Optional

import pandas as pd
import numpy as np
import torch

# paragraph spliting
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

import re

# define elements to split on clauses
ABBREVIATIONS = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Jr.', 'Sr.', 'e.g.', 'i.e.', 'U.S.', 'U.K.']
PLACEHOLDERS = {abbr: f"__{i}__" for i, abbr in enumerate(ABBREVIATIONS)}
PATTERN = r'(?<=[.!?,])\s+'


def tokenize_sentence_part(text: str) -> List[str]:
    """
    Tokenize text into sentence_part.

    Parameters
    ----------
    text
        text to tokenize

    Returns
    -------
    sentence_part
        list of sentence_part
    """

    # replace abbreviations with placeholders to ignore the dot in the abbreviation
    for abbr, placeholder in PLACEHOLDERS.items():
        text = text.replace(abbr, placeholder)

    # split the text based on the pattern
    sentences = re.split(PATTERN, text)

    # replace placeholders back to original abbreviations
    sentences = [sentence for sentence in sentences]
    for i, sentence in enumerate(sentences):
        new_sentence = sentence
        for abbr, placeholder in PLACEHOLDERS.items():
            new_sentence = new_sentence.replace(placeholder, abbr)
        sentences[i] = new_sentence

    return sentences


def _split_paragraph(text: str, granularity: str = "sentence"):
    """
    Split text paragraph into smaller part depending on the granularity.

    Parameters
    ----------
    text
        text to split
    granularity
        granularity of split. One of "sentence", "word", "sentence-part"
    """
    if granularity == "sentence":
        return nltk.tokenize.sent_tokenize(text)
    elif granularity == "word":
        return nltk.tokenize.word_tokenize(text)
    elif granularity == "sentence-part":
        return tokenize_sentence_part(text)
    # elif granularity == "clause":
    #     return extract_clauses(text)
    else:
        raise ValueError(f"Unknown granularity {granularity}."\
                         +"It should be one of 'sentence', 'word', 'sentence-part'")


def split_paragraphs(paragraphs: List[str], granularity: str = "sentence"):
    """
    Split list of paragraphs into smaller part depending on the granularity.
    The splitted parts are returned in a unique list.

    Parameters
    ----------
    paragraphs
        list of text to split
    granularity
        granularity of split. One of "sentence", "word", "clause"
    
    Returns
    -------
    splitted_paragraphs
        Unique list of all splitted paragraphs parts, with the same order as the input list.
    """
    if isinstance(paragraphs, str):
        return _split_paragraph(paragraphs, granularity)
    
    return [text
            for paragraph in paragraphs
            for text in _split_paragraph(str(paragraph), granularity)]


def split_paragraph_and_repeat_labels(paragraphs: List[str],
                                      labels: List[int],
                                      granularity: str = "sentence",):
    """
    Split list of paragraphs into smaller part depending on the granularity.
    The splitted parts are returned in a unique list, and the labels are repeated for each part.

    Parameters
    ----------
    paragraphs
        list of text to split
    labels
        list of labels to repeat
    granularity
        granularity of split. One of "sentence", "word", "clause"
    
    Returns
    -------
    splitted_paragraphs
        Unique list of all splitted paragraphs parts, with the same order as the input list.
    """
    if isinstance(paragraphs, pd.DataFrame):
        paragraphs = paragraphs["sentence"]

    if isinstance(paragraphs, str):
        splitted = _split_paragraph(paragraphs, granularity)
        return splitted, [labels[0]] * len(splitted)

    splitted_paragraphs = [_split_paragraph(str(paragraph), granularity) for paragraph in paragraphs]
    repeated_labels = [label
                       for label, splitted_paragraph in zip(labels, splitted_paragraphs)
                       for _ in range(len(splitted_paragraph))]
    flattened_paragraphs = [text for paragraph in splitted_paragraphs for text in paragraph]

    return flattened_paragraphs, repeated_labels

def count_unique_words(paragraphs: List[str],
                       return_counts: bool = False,
                       ratio_min_threshold: int = 0,
                       save_dir: str = None
                       ) -> Union[Dict[str, int], List[str]]:
        """
        Extract unique words from the input paragraphs.
    
        Parameters
        ----------
        paragraphs
            List of paragraphs from which to extract unique words.
        return_counts
            Whether to return the counts of each word or only the unique words.
        ratio_min_threshold
            minimum ratio of occurrences of a word with respect to the number of paragraphs to keep it.
        save_dir
            path to save the unique words.
    
        Returns
        -------
        unique_words
            unique words extracted from the input paragraphs.
        """
        assert isinstance(paragraphs, list), "The input should be a list of paragraphs."

        # filter words with a count lower than the threshold
        count_min_threshold = len(paragraphs) * ratio_min_threshold

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "words_count.json")
        else:
            save_path = None
    
        if save_path is not None and os.path.exists(save_path):
            # load the unique words
            with open(save_path, "r") as file:
                words_count = Counter(json.load(file))
        else:
            lemmatizer = WordNetLemmatizer()
            nltk.download('wordnet')
            stop_words = set(stopwords.words("english")) + ['%', "'", '*', '/']
            determiners = {"the", "a", "an"}

            # Combine stopwords and determiners to filter out during processing
            stop_words.update(determiners)

            # Concatenate all paragraphs and remove punctuation
            text = " ".join(paragraphs)
            text = re.sub(r"[^\w\s]", "", text)

            # Tokenize the entire concatenated text
            words = word_tokenize(text)
    
            # Initialize a Counter to store lemmatized word counts
            words_count = Counter()

            # Process each token: filter out stopwords, lemmatize, and count
            for word in tqdm(words):
                if word not in stop_words:
                    lemma = lemmatizer.lemmatize(word)
                    words_count[lemma] += 1
            
            if save_path is not None:
                # save the unique words
                with open(save_path, "w") as file:
                    json.dump(words_count, file)

        # assert False
        words_count_subset = Counter({key: count
                                     for key, count in words_count.items()
                                     if count > count_min_threshold})

        if return_counts:
            return words_count_subset

        return sorted(list(words_count_subset.keys()))


def compute_concepts_activating_words(words: List[str],
                                      activations: torch.tensor,
                                      topk: int = 1,
                                      ) -> Dict[str, Dict[str, float]]:
    """
    Compute the activations of the words and clauses of the input paragraphs.

    Parameters
    ----------
    paragraphs
        list of paragraphs to split
    activations
        Concept activations of the paragraphs a tensor of shape (n_words, n_concepts)
    topk
        number of top and bottom words to keep.
    
    Returns
    -------
    words_activations
        Words activating each concepts. Dict of shape {concept_i: {word_j: activation_ij}}.
        Limited to 2 * topk words.
    """
    are_concepts_non_zeros = (activations != 0).any(axis=0)

    # (n_non_zero_concepts, n_words)
    activations = activations[:, are_concepts_non_zeros].T

    non_zero_concepts_ids = [f"concept_{i}"
                             for i in range(len(are_concepts_non_zeros))
                             if are_concepts_non_zeros[i]]

    # sort activations to get the top and bottom words
    sorted_ids_correspondence = torch.argsort(activations, dim=1)

    concepts_activating_words = {}
    # iterate on concepts
    for concept_id, activation, correspondence in zip(non_zero_concepts_ids, activations, sorted_ids_correspondence):
        # select top activations and corresponding words
        top_ids = reversed(correspondence[-topk:])
        top_activations = activation[top_ids]

        # select bottom activations and corresponding words
        bottom_ids = correspondence[:topk]
        bottom_activations = activation[bottom_ids]

        # for each concept compute the most and least activating words
        concepts_activating_words[concept_id] = {
            "aligned": {
                words[id]: round(activation.item(), 3)
                for id, activation in zip(top_ids, top_activations)
            },
            "opposed": {
                words[id]: round(activation.item(), 3)
                for id, activation in zip(bottom_ids, bottom_activations)
                if round(activation.item(), 3) != 0.0
            }
        }

    return concepts_activating_words