"""
This file contains classes for concept decomposition.

The encode and decode functions were inspired by the transform and inverse_transform
functions of sklearn's decomposition classes.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional, List

import numpy as np
import pickle
import scipy
import torch
from torch import nn

from sklearn.decomposition import FastICA, NMF, PCA, TruncatedSVD


class ConceptEncoderDecoder(nn.Module, ABC):
    """
    Class for concept encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: Optional[str] = None,
                 force_recompute: bool = False):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_features = n_features if n_features is not None else A.shape[1]

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encode(x)
        x = self.decode(x_encoded)
        return x
    
    @property
    @abstractmethod
    def is_differentiable(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def concepts_base(self):
        raise NotImplementedError
    
    # @classmethod
    # @abstractmethod
    # def aggregate(cls, encoder_decoder_list: list['ConceptEncoderDecoder']):
    #     raise NotImplementedError


class ICAEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for ICA encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: Optional[str] = None,
                 force_recompute: bool = False):
        # initialize method
        super().__init__(A=A, n_concepts=n_concepts, n_features=n_features)
        self.max_iter = 500
        self.save_path = os.path.join(save_path, "weights.pt")

        # initialize ICA layers
        self.scale = nn.Linear(self.n_features, self.n_features, bias=True)
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=True)

        # compute ICA
        if self.save_path is not None:
            if not force_recompute:
                try:
                    self.load_state_dict(torch.load(self.save_path))
                except FileNotFoundError:
                    force_recompute = True
        else:
            force_recompute = True
            
        if force_recompute:
            assert A is not None, "Concepts decomposition cannot be computed without input data."

            A = A.numpy()
            ica = FastICA(n_components=self.n_concepts, random_state=random_state, max_iter=self.max_iter)
            ica.fit(A)

            #set layers weights with ICA components
            # set encode scaling
            # X - ica.means_
            self.scale.weight.data = torch.eye(self.n_features)
            self.scale.bias.data = - torch.tensor(ica.mean_)

            # set encode projection
            # X_scaled @ ica.components_.T
            self.project.weight.data = torch.tensor(ica.components_)

            # set decode
            # X @ ica.mixing_.T + ica.means_
            self.reconstruct.weight.data = torch.tensor(ica.mixing_)
            self.reconstruct.bias.data = torch.tensor(ica.mean_)

            if self.save_path is not None:
                torch.save(self.state_dict(), self.save_path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (X - ica.means_) @ ica.components_.T
        x_scaled = self.scale(x)
        x_projected = self.project(x_scaled)
        return x_projected

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)
        
    @property
    def is_differentiable(self):
        return True
    
    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class NMFEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for NMF encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: Optional[str] = None,
                 force_recompute: bool = False):
        # initialize method
        super().__init__(A=A, n_concepts=n_concepts, n_features=n_features)
        self.max_iter = 500
        self.save_path = save_path
        self.torch_save_path = os.path.join(self.save_path, "torch_weights.pt")
        self.sklearn_save_path = os.path.join(self.save_path, "sklearn_components.pkl")

        # initialize NMF layers
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=False)

        # compute NMF
        self.nmf = NMF(n_components=self.n_concepts, random_state=random_state, max_iter=self.max_iter)

        if self.save_path is not None:
            if not force_recompute:
                try:
                    self.load_state_dict(torch.load(self.torch_save_path))
                    components = pickle.load(open(self.sklearn_save_path, "rb"))
                    self.nmf.components_ = components
                    self.nmf.n_features_in_ = self.n_features
                except FileNotFoundError:
                    force_recompute = True
        else:
            force_recompute = True
            
        if force_recompute:
            assert A is not None, "Concepts decomposition cannot be computed without input data."
            
            # input matrix should be positive
            assert (A >= 0).all(), "NMF only works with non-negative data."

            A = A.cpu().numpy()
            self.nmf.fit(A)

            # set decode weights
            # Xt @ nmf.components_
            self.reconstruct.weight.data = torch.tensor(self.nmf.components_.T)

            if self.save_path is not None:
                torch.save(self.state_dict(), self.torch_save_path)

                # save nmf components
                pickle.dump(self.nmf.components_, open(self.sklearn_save_path, "wb"))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        concepts = self.nmf.transform(x.cpu().numpy())
        return torch.tensor(concepts).to(x.device)
    
    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return False
    
    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class PCAEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for PCA encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: Optional[str] = None,
                 force_recompute: bool = False):
        # initialize method
        super().__init__(A=A, n_concepts=n_concepts, n_features=n_features)
        self.save_path = os.path.join(save_path, "weights.pt")

        # initialize PCA layers
        self.scale = nn.Linear(self.n_features, self.n_features, bias=True)
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=True)

        # compute PCA
        if self.save_path is not None:
            if not force_recompute:
                try:
                    self.load_state_dict(torch.load(self.save_path))
                except FileNotFoundError:
                    force_recompute = True
        else:
            force_recompute = True
            
        if force_recompute:
            assert A is not None, "Concepts decomposition cannot be computed without input data."

            A = A.numpy()
            pca = PCA(n_components=self.n_concepts, random_state=random_state)
            pca.fit(A)

            # set weights
            # set encode scaling
            # X - pca.mean_
            self.scale.weight.data = torch.eye(self.n_features)
            self.scale.bias.data = - torch.tensor(pca.mean_)

            # set encode projection
            # X_scaled @ pca.components_.T
            self.project.weight.data = torch.tensor(pca.components_)
            
            # set decode
            # X @ pca.components_ + pca.mean_
            self.reconstruct.weight.data = torch.tensor(pca.components_.T)
            self.reconstruct.bias.data = torch.tensor(pca.mean_)

            if self.save_path is not None:
                torch.save(self.state_dict(), self.save_path)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (X - pca.mean_) @ pca.components_.T 
        x_scaled = self.scale(x)
        return self.project(x_scaled)
    
    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)
    
    @property
    def is_differentiable(self):
        return True
    
    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class SVDEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for SVD encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    The concept space is given by U @ Sigma.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: Optional[str] = None,
                 force_recompute: bool = False):
        # initialize method
        super().__init__(A=A, n_concepts=n_concepts, n_features=n_features)
        self.save_path = os.path.join(save_path, "weights.pt")

        # initialize SVD layers
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=False)

        # compute SVD
        if self.save_path is not None:
            if not force_recompute:
                try:
                    self.load_state_dict(torch.load(self.save_path))
                except FileNotFoundError:
                    force_recompute = True
        else:
            force_recompute = True
            
        if force_recompute:
            assert A is not None, "Concepts decomposition cannot be computed without input data."

            A = A.numpy()
            svd = TruncatedSVD(n_components=self.n_concepts, random_state=random_state)
            svd.fit(A)

            #set weights
            # set encode
            # X @ svd.components_.T
            self.project.weight.data = torch.tensor(svd.components_)

            # set decode
            # X @ svd.components_
            self.reconstruct.weight.data = torch.tensor(svd.components_.T)

            if self.save_path is not None:
                torch.save(self.state_dict(), self.save_path)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return True
    
    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T