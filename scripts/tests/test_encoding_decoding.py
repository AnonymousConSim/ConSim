import os
import shutil
from time import time
from tqdm import tqdm

import torch
from torch.nn import functional as F
import numpy as np
from scipy.optimize import minimize

from concept_encoder_decoder import (ICAEncoderDecoder, NMFEncoderDecoder, PCAEncoderDecoder,
                                     SparseSVDEncoderDecoder, SVDEncoderDecoder)

# from concept_encoder_decoder import KMeansEncoderDecoder

from sparse_autoencoder import SparseAutoEncoder


def test_decompositions_encoding_decoding():
    """
    For all decompositions, compute concepts encoding and decoding, then check the error.
    """
    n_concepts_possibilities = [3, 5, 10, 20, 50]  # 150 is too slow

    embeddings_path = "/datasets/shared_datasets/BIOS/models/distilbert-base-uncased_lr1e-05_epochs1_maxlen128_batch32_positive/embeddings"
    train_embeddings = torch.load(os.path.join(embeddings_path, "train_embeddings.pt"), map_location=torch.device("cpu"))
    val_embeddings = torch.load(os.path.join(embeddings_path, "val_embeddings.pt"), map_location=torch.device("cpu"))
    print(f"Embeddings shape: train: {train_embeddings.shape}, val: {val_embeddings.shape}")

    encoder_decoders = {
        # "KMeans": KMeansEncoderDecoder,  # KMeans inverse_transform does not exist, thus decoding is not possible
        "ICA": ICAEncoderDecoder,
        "NMF": NMFEncoderDecoder,
        "PCA": PCAEncoderDecoder,
        "SparseSVD": SparseSVDEncoderDecoder,
        "SVD": SVDEncoderDecoder,
    }

    print(f"\nBaseline")
    train_mean = train_embeddings.mean(dim=0, keepdims=True)
    repeated_train_mean = train_mean.repeat(val_embeddings.shape[0], 1)
    baseline_val_error = F.mse_loss(val_embeddings, repeated_train_mean)
    print(f"\tval error: {round(float(baseline_val_error), 5)}")

    for encoder_decoder_name, encoder_decoder_class in encoder_decoders.items():
        print(f"\nEncoderDecoder: {encoder_decoder_name}")
        for n_concepts in n_concepts_possibilities:
            t0 = time()
            cpt_enc_dec = encoder_decoder_class(train_embeddings, n_concepts=n_concepts, random_state=42)
            init_time = time() - t0

            encoded_concepts = cpt_enc_dec.encode(train_embeddings)
            decoded_encoded_embeddings = cpt_enc_dec.decode(encoded_concepts)

            
            t1 = time()
            reconstructed_embeddings = cpt_enc_dec(val_embeddings)
            encode_decode_time = time() - t1

            # check the errors
            decoding_error = F.mse_loss(encoded_concepts, train_embeddings)
            encoding_decoding_error = F.mse_loss(decoded_encoded_embeddings, train_embeddings)
            reconstruction_error = F.mse_loss(reconstructed_embeddings, val_embeddings)

            print(f"\tn_concepts: {n_concepts}, "+\
                  f"decoding error: {round(float(decoding_error), 5)}, "+\
                  f"train error: {round(float(encoding_decoding_error), 5)}, "+\
                  f"val error: {round(float(reconstruction_error), 5)}, "+\
                  f"init time: {round(init_time, 3)}s, "+\
                  f"encoding-decoding time: {round(encode_decode_time, 3)}s")


def test_sae_encoding_decoding():
    """
    For all types of autoencoders, compute the penrose right inverse and check the error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_concepts_possibilities = [3, 5, 10, 20, 50, 150, 500]

    # embeddings_path = "/datasets/shared_datasets/BIOS/models/distilbert-base-uncased_lr1e-05_epochs1_maxlen128_batch32_positive/embeddings"
    embeddings_path = "/datasets/shared_datasets/BIOS/models/distilbert-base-uncased_lr1e-05_epochs1_maxlen128_batch32/embeddings"
    train_embeddings = torch.load(os.path.join(embeddings_path, "train_embeddings.pt"), map_location=torch.device("cpu"))
    val_embeddings = torch.load(os.path.join(embeddings_path, "val_embeddings.pt"), map_location=torch.device("cpu"))

    print(f"Embeddings shape: train: {train_embeddings.shape}, val: {val_embeddings.shape}")

    print(f"\nBaseline")
    train_mean = train_embeddings.mean(dim=0, keepdims=True)
    repeated_train_mean = train_mean.repeat(val_embeddings.shape[0], 1)
    baseline_val_error = F.mse_loss(val_embeddings, repeated_train_mean)
    print(f"\tval error: {round(float(baseline_val_error), 5)}")

    val_embeddings = val_embeddings.to(device)

    activations = ["binary", "bounded", "positive_bounded", "positive", "any"]  #   # "ternary", 

    for activation in activations:
        print(f"\nSparseAutoEncoderActivation: {activation}")
        for n_concepts in n_concepts_possibilities:
            t0 = time()
            sae = SparseAutoEncoder(train_embeddings, n_concepts=n_concepts, concept_space=activation, random_state=42,
                                    encoder_layers=[], l1_coef=0.001, epochs=500)  # max(50, 2 * n_concepts)
            init_time = time() - t0

            sae = sae.to(device)
            with torch.no_grad():
                l0 = (sae.encode(val_embeddings) != 0).type(torch.float32).mean()

                t1 = time()
                reconstructed_embeddings = sae(val_embeddings)
                encode_decode_time = time() - t1

                # check the errors
                reconstruction_error = F.mse_loss(reconstructed_embeddings, val_embeddings)

            print(f"\tn_concepts: {n_concepts}, "+\
                  f"val reconstruction error: {round(float(reconstruction_error), 5)}, "+\
                  f"l0: {round(float(l0), 4)}, "+\
                  f"init time: {round(init_time, 3)}s, "+\
                  f"encoding-decoding time: {round(encode_decode_time, 5)}s")


def test_decomposition_saving_loading():
    """
    Test saving and loading of encoder-decoder.
    """
    embeddings_path = "/datasets/shared_datasets/BIOS/models/distilbert-base-uncased_lr1e-05_epochs1_maxlen128_batch32_positive/embeddings"
    train_embeddings = torch.load(os.path.join(embeddings_path, "train_embeddings.pt"), map_location=torch.device("cpu"))
    val_embeddings = torch.load(os.path.join(embeddings_path, "val_embeddings.pt"), map_location=torch.device("cpu"))

    encoder_decoders = {
        # "KMeans": KMeansEncoderDecoder,  # KMeans inverse_transform does not exist, thus decoding is not possible
        "ICA": ICAEncoderDecoder,
        "NMF": NMFEncoderDecoder,
        "PCA": PCAEncoderDecoder,
        "SparseSVD": SparseSVDEncoderDecoder,
        "SVD": SVDEncoderDecoder,
    }

    os.makedirs("temp", exist_ok=True)

    for encoder_decoder_name, encoder_decoder_class in encoder_decoders.items():
        n_concepts = 10

        save_path = os.path.join("temp", f"{encoder_decoder_name}")
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            cpt_enc_dec = encoder_decoder_class(train_embeddings, n_concepts=n_concepts,
                                                save_path=save_path, force_recompute=True)

            encoded_concepts = cpt_enc_dec.encode(val_embeddings)
            reconstructed_embeddings = cpt_enc_dec.decode(encoded_concepts)

            new_cpt_enc_dec = encoder_decoder_class(train_embeddings, n_concepts=n_concepts, save_path=save_path)

            new_encoded_concepts = new_cpt_enc_dec.encode(val_embeddings)
            new_reconstructed_embeddings = new_cpt_enc_dec.decode(new_encoded_concepts)

            assert torch.allclose(encoded_concepts, new_encoded_concepts, atol=1e-6)
            assert torch.allclose(reconstructed_embeddings, new_reconstructed_embeddings, atol=1e-6)

    # empty temp directory 
    shutil.rmtree("temp")


def test_sae_saving_loading():
    """
    Test saving and loading of autoencoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_path = "/datasets/shared_datasets/BIOS/models/distilbert-base-uncased_lr1e-05_epochs1_maxlen128_batch32_positive/embeddings"
    train_embeddings = torch.load(os.path.join(embeddings_path, "train_embeddings.pt"), map_location=torch.device("cpu"))
    val_embeddings = torch.load(os.path.join(embeddings_path, "val_embeddings.pt"), map_location=torch.device("cpu"))
    val_embeddings = val_embeddings.to(device)

    activations = ["binary", "bounded", "positive_bounded", "positive", "any"]  # "ternary", 

    os.makedirs("temp", exist_ok=True)

    for activation in activations:
        n_concepts = 10

        save_path = os.path.join("temp", f"{activation}")
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "torch")

        sae = SparseAutoEncoder(train_embeddings, n_concepts=n_concepts, concept_space=activation,
                                save_path=save_path, force_recompute=True)

        new_sae = SparseAutoEncoder(train_embeddings, n_concepts=n_concepts, concept_space=activation, save_path=save_path)

        with torch.no_grad():
            sae = sae.to(device)
            encoded_concepts = sae.encode(val_embeddings)
            reconstructed_embeddings = sae.decode(encoded_concepts)

            new_sae = new_sae.to(device)
            new_encoded_concepts = new_sae.encode(val_embeddings)
            new_reconstructed_embeddings = new_sae.decode(new_encoded_concepts)

            assert torch.allclose(encoded_concepts, new_encoded_concepts)
            assert torch.allclose(reconstructed_embeddings, new_reconstructed_embeddings)

    # empty temp directory 
    shutil.rmtree("temp")
        


if __name__ == "__main__":
    test_decompositions_encoding_decoding()
    test_sae_encoding_decoding()
    test_decomposition_saving_loading()
    test_sae_saving_loading()
