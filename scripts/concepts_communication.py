# built-in imports
from itertools import product
import json
import os

# libraries imports
import pandas as pd
import torch

# local imports
from utils.datasets_utils import load_split_dataset
from utils.general_utils import get_args
from utils.models_configs import model_name_from_config, get_splitted_model
from utils.models_inference import get_unique_words_embeddings, get_concepts_examples_embeddings
from utils.text_utils import count_unique_words, compute_concepts_activating_words
from concept_encoder_decoder import ICAEncoderDecoder, NMFEncoderDecoder, PCAEncoderDecoder, SVDEncoderDecoder
from sparse_autoencoder import SparseAutoEncoder


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


N_CONCEPTS = [3, 5, 10, 20, 50, 150, 500]

ENCODERS_DECODERS = {
    # "KMeans": KMeansEncoderDecoder,  # KMeans inverse_transform does not exist, thus decoding is not possible
    "ICA": ICAEncoderDecoder,
    "NMF": NMFEncoderDecoder,
    "PCA": PCAEncoderDecoder,
    # "SparseSVD": SparseSVDEncoderDecoder,  # U are the concepts
    "SVD": SVDEncoderDecoder,  # U @ Sigma are the concepts
    "SAE": SparseAutoEncoder,
}

ENCODERS_DECODERS = {  # TODO: remove
    # "KMeans": KMeansEncoderDecoder,  # KMeans inverse_transform does not exist, thus decoding is not possible
    "ICA": ICAEncoderDecoder,
    # "NMF": NMFEncoderDecoder,
    # "PCA": PCAEncoderDecoder,
    # # "SparseSVD": SparseSVDEncoderDecoder,  # U are the concepts
    # "SVD": SVDEncoderDecoder,  # U @ Sigma are the concepts
    # "SAE": SparseAutoEncoder,
}

# concepts activating words
TOPK = 5
RATIO_THRESHOLD = 0.002


def main():
    dataset_config, model_config, args = get_args()
    force_regeneration = args.force
    encoding_batch_size = 2048 * 8

    # forcing concepts from dataset extended by sentence parts
    concepts_dir = "concepts_sentence-part"

    model_name = model_name_from_config(model_config)
    concepts_path = os.path.join(os.getcwd(), "data", concepts_dir, dataset_config.name, model_name)
    
    # load model and tokenizer
    model, tokenizer = get_splitted_model(model_config, dataset_config, include_base=False, device=DEVICE)


    with torch.no_grad():
        # ----------------------------
        # for concepts communication 1
        # load dataset
        train_data, _, _, _, _, _ = load_split_dataset(dataset_config)
        if isinstance(train_data, pd.DataFrame):
            train_data = list(train_data["sentence"])
        
        # get unique words
        unique_words = count_unique_words(
            paragraphs=train_data,
            return_counts=False,
            ratio_min_threshold=RATIO_THRESHOLD,
            save_dir=dataset_config.path,
        )

        # get unique words embeddings
        unique_words_embeddings = get_unique_words_embeddings(
            dataset_path=dataset_config.path,
            model_config=model_config,
            regenerate=False,
            model=model,
            tokenizer=tokenizer,
            max_length=model_config.max_length,
            unique_words=unique_words,
            batch_size=model_config.batch_size if model_config is not None else 1,
            device=DEVICE,
            tqdm_disable=False,
        )
        
        # ----------------------------
        # for concepts communication 2
        o1_concepts_embeddings_path = os.path.join(os.getcwd(), "data", concepts_dir, dataset_config.name, "o1_concepts")

        # get o1 concepts embeddings
        o1_concepts_embeddings = get_concepts_examples_embeddings(
            save_path=o1_concepts_embeddings_path,
            model_name=model_name,
            regenerate=False,
            model=model,
            tokenizer=tokenizer,
            max_length=model_config.max_length,
            batch_size=model_config.batch_size if model_config is not None else 1,
            device=DEVICE,
        )
        n_features = unique_words_embeddings.shape[1]
        # model.del_base_model()
        model.cpu()

        # iterate on concept encoders decoders
        for method_name, n_concepts in product(ENCODERS_DECODERS.keys(), N_CONCEPTS):

            path = os.path.join(concepts_path, method_name, f"{n_concepts}_concepts")
            if not os.path.exists(path) or len(os.listdir(path)) == 0:
                print(f"Concepts not found for {method_name} with {n_concepts} concepts.")
                continue

            print(f"{dataset_config.name} - {model_name} - {concepts_dir}",
                    f"Computing words and o1 concepts that activate concepts for {method_name} with {n_concepts} concepts.")
            
            # concepts activating words path
            concept_activating_words_path = os.path.join(path, "concept_activating_words.json")
            o1_concepts_correspondence_path = os.path.join(path, "o1_concepts_correspondence.json")

            if not force_regeneration\
                    and os.path.exists(concept_activating_words_path)\
                    and os.path.exists(o1_concepts_correspondence_path):
                print(f"\tConcepts activating words and concepts correspondence already computed.")
                continue

            # load concepts encoder decoder
            try:
                concepts_encoder_decoder = ENCODERS_DECODERS[method_name](n_concepts=n_concepts,
                                                                          n_features=n_features,
                                                                          save_path=path)
            except:
                print(f"Error loading concepts encoder decoder {method_name} with {n_concepts} concepts. The method may not exist.")
                continue

            # compute concepts activating words
            # load test embeddings on device
            model.eval()
            model.end_model_to(DEVICE)
            model.end_model_to(torch.float32)
            concepts_encoder_decoder.eval()
            concepts_encoder_decoder.to(DEVICE)

            # ------------------------
            # concepts communication 1
            if not os.path.exists(concept_activating_words_path) or force_regeneration:
                # compute concepts activations
                words_concepts_activations = []
                for i in range(0, len(unique_words_embeddings), encoding_batch_size):
                    # take batch
                    embeddings_batch = unique_words_embeddings[i:i+encoding_batch_size]

                    # infer batch
                    batch_concepts = concepts_encoder_decoder.encode(embeddings_batch.to(DEVICE))

                    # stock result
                    words_concepts_activations.append(batch_concepts.cpu())
                
                # concat results
                words_concepts_activations = torch.cat(words_concepts_activations, dim=0)

                # compute concepts activating words
                concept_activating_words = compute_concepts_activating_words(
                    words=unique_words,
                    activations=words_concepts_activations,
                    topk=TOPK,
                )

                # for concept_id, activating_words in concept_activating_words.items():
                #     print(f"\tconcept {concept_id}: topk words and activations: {activating_words}")

                with open(concept_activating_words_path, 'w') as json_file:
                    json.dump(concept_activating_words, json_file, indent=4)
            
            # ------------------------
            # concepts communication 2
            if not os.path.exists(o1_concepts_correspondence_path) or force_regeneration:
                # compute concepts activations for each o1 concept
                o1_concepts_mean_activations = torch.zeros((len(o1_concepts_embeddings), n_concepts))
                for o1_concept_id, o1_concept_embeddings in enumerate(o1_concepts_embeddings.values()):
                    o1_concepts_mean_activations[o1_concept_id] =\
                        concepts_encoder_decoder.encode(o1_concept_embeddings.to(DEVICE)).mean(dim=0).cpu()
                
                inputs_concepts_activations = compute_concepts_activating_words(
                    words=list(o1_concepts_embeddings.keys()),
                    activations=o1_concepts_mean_activations,
                    topk=1,
                )

                with open(o1_concepts_correspondence_path, 'w') as json_file:
                    json.dump(inputs_concepts_activations, json_file, indent=4)


if __name__ == "__main__":
    main()
