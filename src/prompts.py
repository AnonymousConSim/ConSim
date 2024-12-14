from typing import Dict, Tuple, List, NamedTuple

import numpy as np

class PromptSetting(NamedTuple):
    # global
    anonymous_classes: bool = False
    concepts_activating_words: bool = False
    concepts_global_importances: bool = False

    # learning phase
    lr_samples: bool = False
    lr_concepts_local_contributions: bool = False
    lr_labels: bool = False

    # inference
    inf_samples: bool = True
    inf_concepts_local_contributions: bool = False

    # prediction
    # pred_concepts: bool = False


def setting_to_prompt(
        setting: PromptSetting,
        sentences: List[str],
        predictions: List[float],
        classes: List[str],
        concepts_activating_words: Dict[str, Dict[str, float]],
        classes_concepts_importance: Dict[str, Dict[str, float]],
        sentences_concepts_importances: List[Dict[str, float]],
        ) -> str:
    """
    Create a prompt for the LLM model.
    It adapts to the setting to cover all possibilities

    Parameter
    ---------
    setting: PromptSetting
        Configuration, it says which elements should be included in the prompt.
    sentences: List[str]
        The sentences, the first half serve as examples and the second half is to be classified.
    predictions: List[float]
        The predictions of the model on the sentences.
    classes: List[str]
        The classes of the dataset.
    concepts_activating_words: Dict[str, Dict[str, float]]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: Dict[str, Dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    sentences_concepts_importances
        The importance of concepts for each sentence.
        A list with each element corresponding to one sentence.
        Each element of the list if a dictionary with an importance associated to a concept id.

    Returns
    -------
    prompt: str
        The prompt for the LLM.
    """
    system_prompt_parts = []
    user_prompt_parts = []

    # ==============================================================================================
    # Global
    # ----------------
    # task description

    task_description_prompt = "You are a classifier. For each sample, you have to predict the class. "
    if setting.concepts_activating_words or setting.concepts_global_importances:
        task_description_prompt += "To complete the task, you will be given the concepts and their importance for each class. "
    if setting.lr_samples and setting.lr_labels:
        if setting.lr_concepts_local_contributions:
            task_description_prompt += "You will have examples of samples, labels, and concepts contributions to labels as reference for the task. "
        else:
            task_description_prompt += "You will have examples of samples and labels as reference for the task. "
    if setting.inf_concepts_local_contributions:
        task_description_prompt += "At inference time, you will have concepts contributions to labels. "
    task_description_prompt += "Each sample class prediction should be in the format: 'Sample_{i}: {predicted_class}'."

    assert len(task_description_prompt) > 0
    system_prompt_parts.append(task_description_prompt)

    # -------
    # classes
    # if setting.pred_concepts:
    #     # show the concepts that could be predicted
    #     classes_prompt = f"The concepts are: [{', '.join(concepts_activating_words.keys())}]"
    if setting.anonymous_classes:
        # show the classes without their names
        anonym_classes = {class_name: f"Class_{i}" for i, class_name in enumerate(classes)}
        classes_prompt = f"The classes are: [{', '.join(anonym_classes.values())}]"
    else:
        # show the classes
        classes_prompt = f"The classes are: [{', '.join(classes)}]"
    system_prompt_parts.append(classes_prompt)
    
    # -------------------------
    # concepts activating words
    if setting.concepts_activating_words:
        # for each concept, show 10 words, 5 that aligns the most and 5 that are the most opposed
        concepts_activating_words_prompt = \
            "For each concept, the most aligned and opposed words are:\n" +\
            "\n".join([f"{concept_id}: aligned: {list(words['aligned'].keys())}] opposed: {list(words['opposed'].keys())}"
                       if len(words['opposed']) else f"{concept_id}: aligned: {list(words['aligned'].keys())}"
                       for concept_id, words in concepts_activating_words.items()])
        system_prompt_parts.append(concepts_activating_words_prompt)
    
    # ---------------------------
    # classes concepts importance
    if setting.concepts_global_importances:
        # show the importance of the concepts for each class
        if setting.anonymous_classes:
            classes_concepts_prompt = \
                "The most important concepts and their importance for each class are:\n" +\
                "\n".join([f"{anonym_classes[class_name]}: {value}" for class_name, value in classes_concepts_importance.items()])
        else:
            classes_concepts_prompt = \
                "The most important concepts and their importance for each class are:\n" +\
                "\n".join([f"{key}: {value}" for key, value in classes_concepts_importance.items()])
        system_prompt_parts.append(classes_concepts_prompt)
    

    # ==============================================================================================
    # Learning phase
    mid_index = len(sentences) // 2
    # -------
    # samples
    if setting.lr_samples:
        # show the samples
        lr_samples_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index)])
        system_prompt_parts.append(lr_samples_prompt)
    
    # ----------------------------
    # concepts local contributions
    if setting.lr_concepts_local_contributions:
        # show the concepts contributions to the samples
        lr_concepts_local_contributions_prompt = "\n".join([
            f"Concepts contributions for Sample_{i}: {sentences_concepts_importances[i]}"
            for i in range(mid_index)
        ])
        system_prompt_parts.append(lr_concepts_local_contributions_prompt)
    
    # ------
    # labels
    if setting.lr_labels:
        # show the labels
        if setting.anonymous_classes:
            lr_labels_prompt = "\n".join([f"Sample_{i}: {anonym_classes[classes[predictions[i]]]}" for i in range(mid_index)])
        else:
            lr_labels_prompt = "\n".join([f"Sample_{i}: {classes[predictions[i]]}" for i in range(mid_index)])
        system_prompt_parts.append(lr_labels_prompt)
    

    # ==============================================================================================
    # Inference
    # -------
    # samples
    if setting.inf_samples:
        # show the samples
        inf_samples_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index, 2 * mid_index)])
        user_prompt_parts.append(inf_samples_prompt)
    
    # ----------------------------
    # concepts local contributions
    if setting.inf_concepts_local_contributions:
        # show the concepts contributions to the samples
        inf_concepts_local_contributions_prompt = "\n".join([
            f"Concepts contributions for Sample_{i}: {sentences_concepts_importances[i]}"
            for i in range(mid_index, 2 * mid_index)
        ])
        user_prompt_parts.append(inf_concepts_local_contributions_prompt)


    # concatenate prompts parts
    system_prompt = "\n\n".join(system_prompt_parts)
    user_prompt = "\n\n".join(user_prompt_parts)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return prompt


def quantize_importances(importance: float) -> str:
    """
    Convert the normalized importances to literals.
    The literals are:
    - "++" for values above 0.3
    - "+" for values between 0.05 and 0.3
    - "-" for values between -0.05 and -0.3
    - "--" for values below -0.3

    Parameters
    ----------
    importance: float
        The importance to convert.

    Returns
    -------
    literals: str
        The literals corresponding to the importances.
    """
    if importance <= -0.3:
        return "--"
    
    if importance <= -0.05:
        return "-"
    
    if importance >= 0.05:
        return "+"
    
    if importance >= 0.3:
        return "++"

    return None


def filter_and_quantize_concepts_importances(
        concepts_activating_words: Dict[str, Dict[str, float]],
        classes_concepts_importance: Dict[str, Dict[str, float]],
        sentences_concepts_importances: np.ndarray,
        importance_threshold: float = 0.05
        ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], np.ndarray]:
    """
    Filter the concepts importance and quantize the values.

    Parameters
    ----------
    concepts_activating_words: Dict[str, Dict[str, float]]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: Dict[str, Dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    sentences_concepts_importances: np.ndarray
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    importance_threshold: float
        The threshold to select the most important concepts for each class.
        The threshold correspond to the cumulative importance of the concepts to keep.
    
    Returns
    -------
    concepts_activating_words: Dict[str, Dict[str, float]]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: Dict[str, Dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    sentences_concepts_importances: np.ndarray
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    """

    # filter concepts which are important for at least one class
    concepts_to_keep = []
    while len(concepts_to_keep) == 0:
        for class_name, concepts_importance in classes_concepts_importance.items():
            if len(concepts_importance) == 0:
                continue

            # normalize the importances
            importances = np.abs(np.array(list(concepts_importance.values())))
            normalized_importances = importances / importances.sum()

            # select the important concepts
            added_concepts = np.where(normalized_importances > importance_threshold)[0]
            concepts_to_keep.extend(added_concepts)
        if len(concepts_to_keep) == 0:
            importance_threshold /= 2

    concepts_to_show = np.unique(concepts_to_keep)
    activating_words_concepts_ids = np.array([int(cpt.split("_")[-1]) for cpt in concepts_activating_words.keys()], dtype=int)
    concepts_to_show = np.intersect1d(concepts_to_show, activating_words_concepts_ids)

    # filter the concepts activating words
    concepts_activating_words = {
        f"concept_{c}": concepts_activating_words[f"concept_{c}"]
        for c in concepts_to_show
    }
    
    # filter the concepts importance
    classes_concepts_importance = {
        class_name: {
            c: quantize_importances(importance)
            for c, importance in concepts_importance.items()
            if int(c.split("_")[-1]) in concepts_to_show and quantize_importances(importance) is not None
        }
        for class_name, concepts_importance in classes_concepts_importance.items()
    }

    # normalize sentences concepts importances
    sentences_concepts_importances = sentences_concepts_importances / np.abs(sentences_concepts_importances).sum(axis=1)[:, np.newaxis]

    # clean elements to leave only the important concepts and quantize values to literals
    filtered_sentences_concepts_contributions = [
        {
            f"concept_{c}": quantize_importances(importance)
            for c, importance in enumerate(sentence_concepts_importances)
            if c in concepts_to_show and quantize_importances(importance) is not None
        }
        for sentence_concepts_importances  in sentences_concepts_importances
    ]

    return concepts_activating_words, classes_concepts_importance, filtered_sentences_concepts_contributions



def make_prompts(
        sentences: List[str],
        predictions: List[float],
        classes: List[str],
        concepts_activating_words: Dict[str, Dict[str, float]],
        classes_concepts_importance: Dict[str, Dict[str, float]],
        sentences_concepts_importances: np.ndarray,
        importance_threshold: float = 0.05
        ) -> Dict[str, str]:
    """
    Create prompts for the LLM model.
    There are three types of prompts:
    - without_explanation: The LLM has to predict the model's prediction for the next sentences.
    - with_concepts_explanations: The LLM has to predict the model's prediction for the next sentences.
    - with_concepts_explanations_and_predictions:
    The LLM has to predict the activations of the concepts and the model's prediction for the next sentences.

    Parameters
    ----------
    sentences: List[str]
        The sentences, the first half serve as examples and the second half is to be classified.
    predictions: List[float]
        The predictions of the model on the sentences.
    classes: List[str]
        The classes of the dataset.
    concepts_activating_words: Dict[str, Dict[str, float]]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: Dict[str, Dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    sentences_concepts_importances: np.ndarray
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    importance_threshold: float
        The threshold to select the most important concepts for each class.
        The threshold correspond to the cumulative importance of the concepts to keep.
    
    Returns
    -------
    prompts_and_inputs: Dict[str, str] 
        The prompts for the LLM, the inputs and expected outputs.
    """

    # filter and quantize the concepts importances
    concepts_activating_words, classes_concepts_importance, filtered_sentences_concepts_contributions = \
        filter_and_quantize_concepts_importances(
            concepts_activating_words,
            classes_concepts_importance,
            sentences_concepts_importances,
            importance_threshold
        )

    # regroup prompting kwargs
    kwargs = {
        "sentences": sentences,
        "predictions": predictions,
        "classes": classes,
        "concepts_activating_words": concepts_activating_words,
        "classes_concepts_importance": classes_concepts_importance,
        "sentences_concepts_importances": filtered_sentences_concepts_contributions,
    }

    # list settings
    experiments_settings = {
        # inputs to outputs
        "L1: no LR baseline":\
            PromptSetting(),
        "E1: concepts without LR":\
            PromptSetting(concepts_activating_words=True, concepts_global_importances=True),
        "L2: with LR baseline":\
            PromptSetting(lr_samples=True, lr_labels=True),
        "E2: concepts with LR":\
            PromptSetting(concepts_activating_words=True, concepts_global_importances=True,
                          lr_samples=True, lr_labels=True),
        "E3: concepts with contributions at LR":\
            PromptSetting(concepts_activating_words=True, concepts_global_importances=True,
                          lr_samples=True, lr_concepts_local_contributions=True, lr_labels=True),
        # inputs and concepts to outputs
        "U1: concepts with contributions at LR and inf":\
            PromptSetting(concepts_activating_words=True, concepts_global_importances=True,
                          lr_samples=True, lr_concepts_local_contributions=True, lr_labels=True,
                          inf_concepts_local_contributions=True),
    }

    experiments_settings.update({
        "-a:".join(xp_name.split(":")): PromptSetting(**setting._asdict())._replace(anonymous_classes=True)
        for xp_name, setting in experiments_settings.items()
    })

    # create the prompt for each setting
    prompts = {
        xp_name: setting_to_prompt(setting, **kwargs)
        for xp_name, setting in experiments_settings.items()
    }

    return prompts