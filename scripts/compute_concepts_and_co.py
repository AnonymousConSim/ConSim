"""
Compute concepts importance for a given dataset and model and all concepts-based methods.

python scripts/compute_importance.py --dataset BIOS --model distilbert --force
"""
# built-in imports
from contextlib import contextmanager
from itertools import product
import json
import os
import shutil
import signal
from time import time

# libraries imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# local imports
from utils.general_utils import get_args
from utils.models_configs import model_name_from_config, get_splitted_model
from utils.models_inference import get_all_embeddings
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

TIMEOUT_DURATION = 600  # 10 minutes


class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """
    Context manager to limit the execution time of a block of code.

    Args:
        seconds (int): Maximum number of seconds the block is allowed to run.

    Raises:
        TimeoutException: If the block execution exceeds the given time limit.

    Example:
        >>> import time
        >>> try:
        ...     with timeout(2):
        ...         time.sleep(3)
        ... except TimeoutException:
        ...     print("Timeout occurred")
        Timeout occurred
    """
    def _handle_timeout(signum, frame):
        raise TimeoutException("Timeout occurred")

    # Set the signal handler for SIGALRM to _handle_timeout
    signal.signal(signal.SIGALRM, _handle_timeout)
    # Schedule the SIGALRM signal to be sent after `seconds` seconds
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def compute_predictions_and_gradients(model, concepts_encoder_decoder, concepts, device="cuda"):
    # setups
    model.eval()
    model.end_model_to(device)
    model.end_model_to(torch.float32)
    model.zero_grad()
    concepts_encoder_decoder.to(device)
    concepts_encoder_decoder.zero_grad()
    concepts = concepts.to(device)
    concepts.requires_grad_(True)

    # forward pass
    embeddings = concepts_encoder_decoder.decode(concepts)
    logits = model.end_model(embeddings)
    predicted_classes = logits.argmax(dim=1)

    # filter only the logits of interest
    classes_logits = logits.gather(dim=1, index=predicted_classes.unsqueeze(1)).squeeze()

    # compute gradients
    gradients = torch.autograd.grad(
        outputs=classes_logits,
        inputs=concepts,
        grad_outputs=torch.ones_like(classes_logits),
    )[0].detach().cpu()

    predicted_classes = predicted_classes.detach().cpu()

    return predicted_classes, gradients


def batch_compute_predictions_and_gradients(model, concepts_encoder_decoder, concepts, batch_size=None, device="cuda"):
    if batch_size is None:
        predicted_classes, gradients =\
            compute_predictions_and_gradients(model, concepts_encoder_decoder, concepts, device=device)
    else:
        predicted_classes = []
        gradients = []
        for i in range(0, len(concepts), batch_size):
            concepts_batch = concepts[i:i+batch_size]
            predicted_classes_batch, gradients_batch = compute_predictions_and_gradients(
                model=model,
                concepts_encoder_decoder=concepts_encoder_decoder,
                concepts=concepts_batch,
                device=device
            )
            predicted_classes.append(predicted_classes_batch)
            gradients.append(gradients_batch)
        predicted_classes = torch.cat(predicted_classes, dim=0)
        gradients = torch.cat(gradients, dim=0)
    return predicted_classes, gradients


def compute_importance(model: torch.nn.Module,
                       concepts_encoder_decoder: torch.nn.Module,
                       concepts: torch.Tensor,
                       classes: list[str],
                       device: torch.device,
                       compute_gradient_x_input: bool = False) -> dict[str, dict[str, float]]:
    # ========================
    # compute all attributions
    # batched forward and backward pass
    predicted_classes, gradients = batch_compute_predictions_and_gradients(
        model=model,
        concepts_encoder_decoder=concepts_encoder_decoder,
        concepts=concepts,
        batch_size=2048,
        device=device
    )

    if compute_gradient_x_input:
        gradients = (gradients.to(device) * concepts).cpu()  # TODO: test

    # convert to numpy
    gradients_np = gradients.numpy()
    predicted_classes = predicted_classes.numpy()

    # ==================================
    # compute global classes importances

    # for each class select indices of samples that are predicted as this class
    classes_indices = [
        [sample_index
         for sample_index in range(concepts.shape[0])
         if predicted_classes[sample_index] == class_index]
        for class_index in range(len(classes))]
    
    # compute the importance of each concept for each class
    classes_concepts_importances = [
        gradients_np[classes_indices[class_index]].mean(axis=0)
        for class_index in range(len(classes))
    ]

    # format the importance
    classes_concepts_importances = {
        f"{classes[class_index]}": {
            f"concept_{concept_index}": importance
            for concept_index, importance in enumerate(classes_concepts_importances[class_index])
        }
        if len(classes_indices[class_index]) > 0
        else {}  # skip classes with no samples
        for class_index in range(len(classes))
    }

    # normalize importances
    classes_concepts_importances = {
        class_name: {
            concept_name: round(float(importance / sum(abs(np.array(list(importances.values()))))), 3)
            for concept_name, importance in importances.items()
        }
        if sum(abs(np.array(list(importances.values())))) != 0
        else {}  # skip classes where importances are all zeros
        for class_name, importances in classes_concepts_importances.items()
    }

    # =============================
    # compute importances quantiles
    quantiles_percentage = torch.linspace(0, 1, 21)
    concepts_importances_quantiles =\
        torch.quantile(gradients, quantiles_percentage, dim=0).T.numpy().tolist()
    concepts_importances_quantiles = {
        f"concept_{concept}": {
            f"quantile_{int(quantile*100)}": value
            for quantile, value in zip(quantiles_percentage, quantile_values)
        }
        for concept, quantile_values in enumerate(concepts_importances_quantiles)
    }

    return classes_concepts_importances, concepts_importances_quantiles


def main():
    dataset_config, model_config, args = get_args()
    force_regeneration = args.force
    # granularity = args.granularity
    # compute_gradient_x_input = args.gradxinput
    # encoding_batch_size = 2048 * 8
    max_nb_samples = 500_000

    # forcing concepts from dataset extended by sentence parts
    granularity = "sentence-part"

    # forcing gradient x input
    compute_gradient_x_input = True

    model_name = model_name_from_config(model_config)

    concepts_path = os.path.join(os.getcwd(), "data", "concepts", dataset_config.name, model_name)

    # load model and tokenizer
    model, _ = get_splitted_model(model_config, dataset_config, include_base=False, device="cpu")

    # get test embeddings
    train_embeddings, _, test_embeddings = get_all_embeddings(dataset_path=dataset_config.path,
                                                              model_config=model_config,
                                                              regenerate=False,)
    
    if granularity in ["sentence", "sentence-part"]:
        concepts_path = concepts_path.replace("/concepts/", f"/concepts_{granularity}/")

        if len(train_embeddings) < max_nb_samples:
            # train_data = split_paragraphs(train_data, granularity)
            train_embeddings_granularity, _, _ = get_all_embeddings(dataset_path=dataset_config.path,
                                                                    model_config=model_config,
                                                                    regenerate=False,
                                                                    granularity=granularity)
            if train_embeddings_granularity is None:
                exit()
            
            train_embeddings = torch.cat([train_embeddings, train_embeddings_granularity], dim=0)
    
    if len(train_embeddings) > max_nb_samples:
        train_embeddings = train_embeddings[:max_nb_samples]

    if train_embeddings is None or test_embeddings is None:
        raise ValueError(f"\nNo embeddings found for {model_name}.")
    
    # load test embeddings on device
    model.eval()
    model.end_model_to(DEVICE)
    model.end_model_to(torch.float32)

    if force_regeneration:
        shutil.rmtree(concepts_path, ignore_errors=True)

    # iterate on concept encoders decoders
    for method_name, n_concepts in product(list(ENCODERS_DECODERS.keys()), N_CONCEPTS):
        print(f"{dataset_config.name} - {model_name.split('-')[0]} - positive={model_name.split('_')[-1] == 'positive'} -",
                f"{granularity} - gradxinput={compute_gradient_x_input} - {method_name} - {n_concepts} concepts")
        
        if method_name == "NMF" and not args.positive:
            print("\tSkipping NMF with positive embeddings.")
            continue

        path = os.path.join(concepts_path, method_name, f"{n_concepts}_concepts")
        os.makedirs(path, exist_ok=True)

        if method_name in ["ICA", "NMF"] and n_concepts > 200:
            print("\tSkipping ICA and NMF with more than 200 concepts, expecting TimeOut.")
            continue

        # quantiles path
        quantiles_path = os.path.join(path, "quantiles.json")
        
        # importances path
        importances_path = os.path.join(path, "importances.json")
        if compute_gradient_x_input:
            importances_path = importances_path.replace(".json", "_x_input.json")

        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            print(f"\tSKIPPING AS WE EXPECT TIMEOUTS")
            continue

        if os.path.exists(importances_path) and os.path.exists(quantiles_path):
            print(f"\tConcepts, quantiles, and importances have already been computed.")
            continue

        # compute concepts
        try:
            with timeout(TIMEOUT_DURATION):
                t0 = time()
                if n_concepts > 100 and model_name == "llama" and not "SAE" in method_name:
                    print("\tReducing the train dataset size for llama to prevent cpu overload.")
                    concepts_encoder_decoder = ENCODERS_DECODERS[method_name](A=train_embeddings[:100_000],
                                                                              n_concepts=n_concepts,
                                                                              save_path=path)
                else:
                    concepts_encoder_decoder = ENCODERS_DECODERS[method_name](A=train_embeddings,
                                                                              n_concepts=n_concepts,
                                                                              save_path=path)
                t = time() - t0
        except TimeoutException:
            print(f"\tTimeout.")
            continue

        with torch.no_grad():
            test_embeddings = test_embeddings.to(DEVICE)
            concepts_encoder_decoder.to(DEVICE)
            test_concepts = concepts_encoder_decoder.encode(test_embeddings)
            reconstructed_embeddings = concepts_encoder_decoder.decode(test_concepts)

            error = F.mse_loss(reconstructed_embeddings, test_embeddings).item()
            l0 = (test_concepts != 0).type(torch.float32).mean()
            test_embeddings = test_embeddings.cpu()

            print(f"\tTime: {t:.3f}s, Reconstruction error: {error:.4f}, L0: {l0:.4f}")

        # compute quantiles
        if os.path.exists(quantiles_path):
            print(f"\tQuantiles have already been computed.")
        else:
            quantiles_percentage = torch.linspace(0, 1, 21).to(DEVICE)
            concepts_quantiles = torch.quantile(test_concepts, quantiles_percentage, dim=0).cpu().numpy().tolist()
            concepts_quantiles = {
                f"concept_{concept}": {
                    f"quantile_{int(quantile*100)}": value
                    for quantile, value in zip(quantiles_percentage, quantile_values)
                }
                for concept, quantile_values in enumerate(concepts_quantiles)
            }
            with open(quantiles_path, "w") as f:
                json.dump(concepts_quantiles, f, indent=4)

        # compute importance
        if os.path.exists(importances_path):
            print(f"\tConcepts and importances already computed.")
        else:
            importance, importances_quantiles = compute_importance(
                model=model,
                concepts_encoder_decoder=concepts_encoder_decoder,
                concepts=test_concepts,
                classes=dataset_config.labels_names,
                device=DEVICE,
                compute_gradient_x_input=compute_gradient_x_input
            )

            # save importance
            with open(importances_path, "w") as f:
                json.dump(importance, f, indent=4)

            # save importance quantiles
            importances_quantiles_path = importances_path.replace(".json", "_quantiles.json")
            with open(importances_quantiles_path, "w") as f:
                json.dump(importances_quantiles, f, indent=4)


if __name__ == "__main__":
    main()
