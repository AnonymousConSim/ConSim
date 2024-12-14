
import torch

from sparse_autoencoder import SparseAutoEncoder


def test_concept_spaces():
    """
    Ensure concept space of SAE are in the correct range.
    """
    n_samples = 10000
    n_features = 1000
    n_concepts = 100

    embeddings = torch.randn(n_samples, n_features).abs()

    activations = ["any", "binary", "bounded", "positive"]

    for activation in activations:
        print("Activation:", activation)
        sae = SparseAutoEncoder(embeddings, n_concepts=n_concepts, concept_space=activation, stock_concepts=True)
        concepts = sae.encode(embeddings)

        print(f"\tmin: {concepts.min()}, mean: {concepts.mean()}, max: {concepts.max()}")

        if activation == "any":
            assert concepts.min() < 0
            assert concepts.max() > 0
        elif activation == "binary":
            assert concepts.min() == 0
            assert concepts.max() == 1
        elif activation == "bounded":
            assert concepts.min() >= 0
            assert concepts.max() <= 1
        elif activation == "positive":
            assert concepts.min() >= 0
            assert concepts.max() > 0


if __name__ == "__main__":
    test_concept_spaces()