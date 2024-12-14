from itertools import product
import json
import os
from tqdm import tqdm
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.general_utils import get_args
from utils.datasets_utils import load_split_dataset
from utils.models_configs import get_splitted_model, model_name_from_config
from utils.models_inference import get_all_embeddings

from prompts import make_prompts

from concept_encoder_decoder import ICAEncoderDecoder, NMFEncoderDecoder, PCAEncoderDecoder, SVDEncoderDecoder
from sparse_autoencoder import SparseAutoEncoder


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


N_CONCEPTS = [3, 5, 10, 20, 50, 150, 500]

ENCODERS_DECODERS = {
    # "KMeans": KMeansEncoderDecoder,  # KMeans inverse_transform does not exist, thus decoding is not possible
    "PCA": PCAEncoderDecoder,  # fastest
    "ICA": ICAEncoderDecoder,
    "NMF": NMFEncoderDecoder,
    # "SparseSVD": SparseSVDEncoderDecoder,  # U are the concepts
    "SVD": SVDEncoderDecoder,  # U @ Sigma are the concepts
    "SAE": SparseAutoEncoder,
}


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_gradients(model, concepts_encoder_decoder, concepts, device="cuda"):
    # setups
    model.eval()
    model.end_model_to(device)
    model.end_model_to(torch.float32)
    model.zero_grad()
    concepts_encoder_decoder.to(device)
    concepts_encoder_decoder.zero_grad()
    concepts = concepts.to(device)
    concepts.requires_grad_(True)

    with torch.enable_grad():
        # forward pass
        embeddings = concepts_encoder_decoder.decode(concepts)
        logits = model.end_model(embeddings)
        predicted_classes = torch.argmax(logits, dim=1)

        # filter only the logits of interest
        classes_logits = logits.gather(dim=1, index=predicted_classes.unsqueeze(1)).squeeze()

    # compute gradients
    gradients = torch.autograd.grad(
        outputs=classes_logits,
        inputs=concepts,
        grad_outputs=torch.ones_like(classes_logits),
    )[0].detach()

    return predicted_classes, gradients


def find_sentences_indices(model: nn.Module,
                           labels: np.ndarray,
                           embeddings: torch.Tensor,
                           class_subset: Union[bool, List[int]] = False,
                           nb_correct: int = 10,
                           nb_mistakes: int = 10,
                           encoding_batch_size: int = 2048,
                           seed: int = 0) -> np.ndarray:
    # # find correct and incorrect predictions
    # correct_indices = []
    # incorrect_indices = []
    # # print("DEBUG: make_GPT_prompts: find_sentences_indices: embeddings", embeddings.shape)
    # for i, (latent_space, label) in tqdm(enumerate(zip(embeddings, labels))):
    #     # if len(correct_indices) == 0 and len(incorrect_indices) == 0:
    #     #     print("DEBUG: make_GPT_prompts: find_sentences_indices: latent_space", latent_space.shape)
    #     output = model.end_model(latent_space.unsqueeze(0))
    #     prediction = torch.argmax(output).item()
    #     if prediction == label:
    #         correct_indices.append(i)
    #     else:
    #         incorrect_indices.append(i)
    
    # Compute batched predictions
    for i in range(0, len(embeddings), encoding_batch_size):
        batch_embeddings = embeddings[i:i+encoding_batch_size].to(DEVICE)
        batch_outputs = model.end_model(batch_embeddings)
        batch_predictions = torch.argmax(batch_outputs, dim=1).cpu().numpy()
        if i == 0:
            predictions = batch_predictions
        else:
            predictions = np.concatenate([predictions, batch_predictions])
    # outputs = model.end_model(embeddings)
    # predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Find the correct and incorrect indices
    correct_indices = np.nonzero(predictions == labels)[0].tolist()
    incorrect_indices = np.nonzero(predictions != labels)[0].tolist()

    # select sentences
    if class_subset:
        correct_indices = [i for i in correct_indices if labels[i] in class_subset]
        incorrect_indices = [i for i in incorrect_indices if labels[i] in class_subset]
    else:
        class_subset = np.unique(labels)

    if len(correct_indices) < nb_correct or len(incorrect_indices) < nb_mistakes:
        raise ValueError(f"Not enough correct or incorrect predictions to select {nb_correct} correct and {nb_mistakes} incorrect.")
    
    # if len(class_subset) > nb_correct or len(class_subset) > nb_mistakes:
    #     raise ValueError(f"Class subset is larger than the number of correct or incorrect, not all classes can be represented.")

    correct_indices = np.array(correct_indices)
    incorrect_indices = np.array(incorrect_indices)

    np.random.seed(seed)
    np.random.shuffle(correct_indices)
    np.random.shuffle(incorrect_indices)

    # select nb_correct correct and nb_mistakes incorrect, each class should be represented
    nb_correct_elements_per_class = nb_correct // len(class_subset)
    nb_mistakes_elements_per_class = nb_mistakes // len(class_subset)

    # select correct and incorrect indices for each class
    class_wise_correct_indices = []
    class_wise_incorrect_indices = []
    for c in class_subset:
        class_wise_correct_indices.append(correct_indices[labels[correct_indices] == c])
        class_wise_incorrect_indices.append(incorrect_indices[labels[incorrect_indices] == c])

    selected_correct_indices = np.concatenate([c[:nb_correct_elements_per_class] for c in class_wise_correct_indices])
    selected_incorrect_indices = np.concatenate([c[:nb_mistakes_elements_per_class] for c in class_wise_incorrect_indices])

    # in case the number of correct or incorrect is not a multiple of the number of classes
    nb_correct_remaining = nb_correct - nb_correct_elements_per_class * len(class_subset)
    if nb_correct_remaining:
        additional_possible_correct_indices = np.concatenate([c[nb_correct_elements_per_class:] for c in class_wise_correct_indices])
        additional_correct_indices = np.random.choice(additional_possible_correct_indices, size=nb_correct_remaining, replace=False)
        selected_correct_indices = np.concatenate([selected_correct_indices, additional_correct_indices])
    
    nb_mistakes_remaining = nb_mistakes - nb_mistakes_elements_per_class * len(class_subset)
    if nb_mistakes_remaining:
        additional_possible_incorrect_indices = np.concatenate([c[nb_mistakes_elements_per_class:] for c in class_wise_incorrect_indices])
        additional_incorrect_indices = np.random.choice(additional_possible_incorrect_indices, size=nb_mistakes_remaining, replace=False)
        selected_incorrect_indices = np.concatenate([selected_incorrect_indices, additional_incorrect_indices])

    indices = np.concatenate([selected_correct_indices, selected_incorrect_indices])

    np.random.shuffle(indices)

    return indices


def main():
    dataset_config, model_config, args = get_args()
    force_regeneration = True  # args.force
    granularity = args.granularity
    gradient_x_input = args.gradxinput
    positive = args.positive
    encoding_batch_size = 2048
    nb_correct = 20  # TODO: increase and compare results
    nb_mistakes = 20  # TODO: increase and compare results
    class_subset = False
    seeds = [0, 1, 2, 3, 4, 5, 6]

    # print("\tForcing activating words from unique words and importance with GradientInput.")
    granularity = "unique-words"
    gradient_x_input = True

    samples_setting = f"{nb_correct}-{nb_mistakes}samples"

    if class_subset:
        class_subset = list(range(dataset_config.num_labels))
        classes = [dataset_config.labels_names[c] for c in class_subset]
    else:
        classes = dataset_config.labels_names
    
    importances_suffix = "importances_x_input.json"
    concepts_communication_suffixes = ["concept_activating_words", "o1_concepts_correspondence"]

    model_name = model_name_from_config(model_config)
    # print("\tUsing concepts computed with the initial dataset completed with splitted sentences.")
    concepts_root = os.path.join(os.getcwd(), "data", "concepts_sentence-part", dataset_config.name, model_name)

    # load dataset
    _, _, _, _, test_data, test_labels = load_split_dataset(dataset_config)

    # load model and tokenizer
    model, _ = get_splitted_model(model_config, dataset_config, include_base=False, device="cpu")

    with torch.no_grad():
        # get test embeddings
        _, _, test_embeddings = get_all_embeddings(dataset_path=dataset_config.path,
                                                   model_config=model_config,
                                                   regenerate=False)
        
        # load test embeddings on device
        model.eval()
        model.end_model_to(DEVICE)
        model.end_model_to(torch.float32)

        n_features = test_embeddings.shape[1]
        
        if dataset_config.name[:4] == "BIOS":
            test_data = test_data["sentence"]

        sentences = list(test_data)
        labels = np.array(test_labels)

        for seed in seeds:
            prompt_path = os.path.join(os.getcwd(), "data", "prompts", samples_setting, dataset_config.name, model_name, f"seed{seed}")
            os.makedirs(prompt_path, exist_ok=True)

            # load are compute sentences indices
            sentences_indices_path = os.path.join(prompt_path, "sentences_indices.npy")
            if os.path.exists(sentences_indices_path) and not force_regeneration:
                sentences_indices = np.load(sentences_indices_path)
            else:
                # select sentences
                sentences_indices = find_sentences_indices(
                    model=model,
                    labels=labels,
                    embeddings=test_embeddings,
                    class_subset=class_subset,
                    nb_correct=nb_correct,
                    nb_mistakes=nb_mistakes,
                    encoding_batch_size=encoding_batch_size,
                    seed=seed
                )
                np.save(sentences_indices_path, sentences_indices)

            assert len(sentences_indices) == nb_correct + nb_mistakes

            # extract data of the selected sentences
            selected_sentences = [sentences[i] for i in sentences_indices]
            selected_sentences_labels = [labels[i] for i in sentences_indices]
            embeddings = test_embeddings[sentences_indices].to(DEVICE)
            predictions = list(model.end_model(embeddings).argmax(dim=1).cpu().numpy())

            # format models predictions and labels to dicts
            sentences_dict = "\n".join([f"Sample_{i}: {selected_sentences[i]}" for i in range(len(selected_sentences))])

            labels_dict = "\n".join([
                f"Sample_{i}: {classes[selected_sentences_labels[i]]}"
                for i in range(len(selected_sentences_labels))
            ])

            f_predictions_dict = "\n".join([
                f"Sample_{i}: {classes[predictions[i]]}"
                for i in range(len(predictions))
                    ])

            # iterate on concept encoders decoders
            for method_name, n_concepts in product(ENCODERS_DECODERS.keys(), N_CONCEPTS):
                if method_name == "NMF" and not positive:
                    continue

                if os.path.exists(os.path.join(prompt_path, f"{method_name}_{n_concepts}concepts_{concepts_communication_suffixes[0]}_prompts.json"))\
                        and os.path.exists(os.path.join(prompt_path, f"{method_name}_{n_concepts}concepts_{concepts_communication_suffixes[1]}_prompts.json"))\
                        and not force_regeneration:
                    # print(f"Prompts already computed for {method_name} with {n_concepts} concepts.")
                    continue

                concepts_path = os.path.join(concepts_root, method_name, f"{n_concepts}_concepts")
                if not os.path.exists(concepts_path) or len(os.listdir(concepts_path)) == 0:
                    print(f"Concepts not found for {method_name} with {n_concepts} concepts.")
                    continue

                if not os.path.exists(os.path.join(concepts_path, importances_suffix))\
                        or not os.path.exists(os.path.join(concepts_path, concepts_communication_suffixes[0] + ".json"))\
                        or not os.path.exists(os.path.join(concepts_path, concepts_communication_suffixes[1] + ".json")):
                    print(f"Missing elements for {method_name} with {n_concepts} concepts.")
                    continue

                # load concepts encoder decoder
                try:
                    encoder_decoder = ENCODERS_DECODERS[method_name](n_concepts=n_concepts,
                                                                     n_features=n_features,
                                                                     save_path=concepts_path)
                except:
                    print(f"Error loading concepts encoder decoder {method_name} with {n_concepts} concepts. The method may not exist.")
                    continue
                encoder_decoder.to(DEVICE)

                # compute concepts activations
                concepts_activations = encoder_decoder.encode(embeddings.to(DEVICE)) 

                # compute concepts predictions and local importances                   
                fc_predictions, sentences_concepts_importances = compute_gradients(model, encoder_decoder, concepts_activations, device=DEVICE)
                if gradient_x_input:
                    sentences_concepts_importances = sentences_concepts_importances * concepts_activations
                
                del concepts_activations
                del encoder_decoder
                sentences_concepts_importances = sentences_concepts_importances.cpu().numpy()

                # format models predictions and labels to dicts
                fc_predictions_dict = "\n".join([
                    f"Sample_{i}: {classes[fc_predictions[i]]}"
                    for i in range(len(predictions))
                ])

                # load importances
                with open(os.path.join(concepts_path, importances_suffix), 'r') as json_file:
                    classes_concepts_importance = json.load(json_file)
                
                # check if there are importances, otherwise, concepts are all zeros
                if sum([len(activations) for activations in classes_concepts_importance.values()]) == 0:
                    # print("\tNo importances, skipping.")
                    continue

                for concepts_communication_suffix in concepts_communication_suffixes:
                    # check if the prompts exist
                    prompt_name = f"{method_name}_{n_concepts}concepts_{concepts_communication_suffix}_prompts.json"
                    prompt_save_path = os.path.join(prompt_path, prompt_name)

                    print(f"{dataset_config.name} - {model_name.split('-')[0]} - positive={model_name.split('_')[-1] == 'positive'} -",
                          f"Seed {seed} - Computing prompts for {method_name} with {n_concepts} concepts.",
                          f"Communication method {concepts_communication_suffix}.")

                    # if os.path.exists(prompt_save_path) and not force_regeneration:
                    #     print("\tPrompts already computed, skipping.")
                    #     continue

                    # load concepts activating words from json
                    with open(os.path.join(concepts_path, concepts_communication_suffix + ".json"), 'r') as json_file:
                        concepts_activating_words = json.load(json_file)

                    # test if there are concepts activating words, otherwise, concepts are all zeros
                    if sum([len(words) for words in concepts_activating_words.values()]) == 0:
                        # print("\tNo activating words, skipping.")
                        continue

                    # create prompts
                    prompts = make_prompts(
                        sentences=selected_sentences,
                        predictions=predictions,
                        classes=classes,
                        concepts_activating_words=concepts_activating_words,
                        classes_concepts_importance=classes_concepts_importance,
                        sentences_concepts_importances=sentences_concepts_importances,
                    )

                    # add data to prompt for saving
                    prompts.update({
                        # "sentences": sentences_dict,
                        "labels": labels_dict,
                        "f_predictions": f_predictions_dict,
                        "fc_predictions": fc_predictions_dict,
                    })

                    # save prompts
                    with open(prompt_save_path, 'w') as json_file:
                        json.dump(prompts, json_file, indent=4)
                    # print(f"\tPrompts saved at {prompt_save_path}.")

                    # for prompt_name, prompt in prompts.items():
                    #     if isinstance(prompt, list):
                    #         print("\n\n\n", prompt_name)
                    #         for name, content in prompt[0].items():
                    #             print(f"{name}: {content}")
                    #         for name, content in prompt[1].items():
                    #             print(f"\n\n{name}: {content}\n")
                    #     else:
                    #         print(f"{prompt_name}:\n{prompt}\n\n\n")


if __name__ == "__main__":
    main()