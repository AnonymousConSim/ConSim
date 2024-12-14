from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset

from concept_encoder_decoder import ConceptEncoderDecoder


class _PositiveBinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(torch.sign(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 0
        return grad_input


class _TernaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        sign_neg = torch.sign(input - 1/3)
        sign_pos = torch.sign(input + 1/3)
        return sign_neg + sign_pos / 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


class _AutoEncoder(nn.Module, ABC):
    def __init__(self,
                 in_features: int,
                 n_concepts: int,
                 encoder_layers: Optional[List[int]] = None):
        super().__init__()

        if encoder_layers is None:
            encoder_layers = []

        # arguments
        self.in_features = in_features
        self.n_concepts = n_concepts

        # pre-encoder bias
        self.pre_encoder_bias = nn.Parameter(torch.zeros(in_features))

        # encoder layers with ReLU in between
        encoders_input_sizes = [in_features] + encoder_layers
        encoders_output_sizes = encoder_layers + [n_concepts]

        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=True)
            for input_size, output_size in zip(encoders_input_sizes, encoders_output_sizes)
        ])

        self.relu = nn.ReLU()

        # decoder
        self.decoder = nn.Linear(self.n_concepts, self.in_features, bias=True)

    @staticmethod
    @abstractmethod
    def activation(self, x):
        pass

    def encode(self, x):
        # pre-encoder bias
        x = x + self.pre_encoder_bias

        # loop ignored if there is only one encoder layer
        for encoder in self.encoder_layers[:-1]:
            x = encoder(x)
            x = self.relu(x)
        
        x = self.encoder_layers[-1](x)
        x = self.activation(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x_encoded = self.encode(x)
        x = self.decode(x_encoded)
        return x


class _AnyAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are unbounded and can be positive or negative.
    """
    def activation(self, x):
        return x


class _PositiveBinaryAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are binary, either 0 or 1.
    """
    def activation(self, x):
        return _PositiveBinaryActivation.apply(x)


class _TernaryAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are binary, either -1, 0, or 1.
    """
    def activation(self, x):
        return _TernaryActivation.apply(x)


class _PositiveBoundedAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are bounded between 0 and 1.
    """
    def activation(self, x):
        return torch.tanh(F.relu(x))


class _BoundedAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are bounded between -1 and 1.
    """
    def activation(self, x):
        return torch.tanh(x)


class _PositiveAutoEncoder(_AutoEncoder):
    """
    Concept coefficients are positive.
    """
    def activation(self, x):
        return F.relu(x)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


class DeadNeuronsResampling:  # TODO: add to autoencoders
    # https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling
    def __init__(self,
                 n_concepts: int,
                 activation_threshold: float = 0.0001,
                 resampling_rate: float = 0.2,
                 resampling_frequency: int = 25000,
                 activity_patience: int = 12500,
                 subset_size_for_loss: int = 1000,
                 device: torch.device = torch.device("cuda")):
        self.threshold = torch.tensor(activation_threshold, dtype=torch.float32).to(device)
        self.resampling_rate = torch.tensor(resampling_rate, dtype=torch.float32).to(device)
        self.resampling_frequency = resampling_frequency
        self.activity_patience = activity_patience
        self.subset_size_for_loss = subset_size_for_loss

        self.step = torch.tensor(0, dtype=torch.float32).to(device)
        self.last_activity = torch.zeros(n_concepts).to(device)
        self.should_resample = False

    def monitor(self, concepts_activation: torch.tensor,):
        self.step += 1
        activated_mask = (concepts_activation.abs() > self.threshold).any(axis=0)
        self.last_activity[activated_mask] = torch.full((activated_mask.sum(),), self.step, device=self.last_activity.device)

        if self.step % self.resampling_frequency == 0:
            self.should_resample = True

    def resample(self, autoencoder: _AutoEncoder, train_loader: DataLoader, device: torch.device):
        with torch.no_grad():
            dead_neurons = self.last_activity < self.step - self.activity_patience
            nb_dead_neurons = dead_neurons.sum().item()

            if nb_dead_neurons > 0:
                # compute the loss on a random subset of inputs
                random_indices = np.random.choice(len(train_loader.dataset),
                                                  self.subset_size_for_loss, replace=False)
                subset = Subset(train_loader.dataset, random_indices)
                subset_loader = DataLoader(subset, batch_size=256, shuffle=True)
                losses = []
                all_features = []
                for features in subset_loader:
                    all_features.append(features)
                    features = features.to(device)
                    concepts = autoencoder.encode(features)
                    reconstructed_embeddings = autoencoder.decode(concepts)
                    losses.append((reconstructed_embeddings - features).pow(2).mean(dim=1))
                all_features = torch.cat(all_features)

                # compute the probability of picking each input to replace dead neurons dictionary vectors
                losses = torch.cat(losses)
                pick_proba = 1 / (losses ** 2 + 1e-6)
                pick_proba = pick_proba / pick_proba.sum()
                picked_indices = np.random.choice(len(subset), nb_dead_neurons, p=pick_proba.cpu().numpy(), replace=False)

                # compute the average norm of alive neurons encoding vectors
                concept_encoding_layers = autoencoder.encoder_layers
                if isinstance(concept_encoding_layers, nn.ModuleList):
                    last_encoding_layer = concept_encoding_layers[-1]
                else:
                    last_encoding_layer = concept_encoding_layers
                decoding_layer = autoencoder.decoder
                alive_neurons = ~dead_neurons
                alive_neurons_encoding = last_encoding_layer.weight[alive_neurons]
                alive_neurons_norm = alive_neurons_encoding.norm(p=2, dim=1).mean()
                
                # resample dead neurons
                if alive_neurons.any():
                    dead_neuron_id = 0
                    for i in range(self.last_activity.shape[0]):
                        if dead_neurons[i]:
                            # Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector for the dead autoencoder neuron.
                            new_dictionary_vector = all_features[picked_indices[dead_neuron_id]]
                            normalized_dictionary_vector = new_dictionary_vector / new_dictionary_vector.norm(p=2)
                            decoding_layer.weight[:, i] = normalized_dictionary_vector

                            # For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder weights for alive neurons × 0.2. Set the corresponding encoder bias element to zero.
                            last_encoding_layer.weight[i] = self.resampling_rate * alive_neurons_norm * last_encoding_layer.weight[i]
                            dead_neuron_id += 1
                else:
                    # All neurons are dead, resampling all neurons
                    last_encoding_layer.weight = nn.Parameter(
                        last_encoding_layer.weight + (self.resampling_rate / 2) * torch.randn_like(last_encoding_layer.weight)
                    ).to(device)

        self.last_activity = torch.full_like(self.last_activity, self.step)
        self.should_resample = False
        

class SparseAutoEncoder(ConceptEncoderDecoder):
    def __init__(self,
                 A: torch.tensor = None,
                 n_concepts: int = 20,
                 n_features: int = None,
                 random_state: int = 0,
                 save_path: str = None,
                 force_recompute: bool = False,
                 concept_space: str = "positive",
                 device: torch.device = torch.device("cuda"),
                 batch_size: int = 256,
                 encoder_layers: list = None,  # if None, then a single linear layer is used
                 train_val_fraction: float = 0.8,
                 nb_steps: int = 100_000,
                 learning_rate: float = 1e-3,  # 1e-3
                 optimizer: str = "Adam",
                 lr_scheduler_step_size: int = 1,
                 lr_scheduler_gamma: float = 0.98,
                 loss: nn.modules.loss._Loss = nn.MSELoss(),
                 l1_coef: float = 0.001,
                 normalize_decoder: bool = True,
                 early_stopping_patience: int = 0.3,
                 show_curves: bool = False):
        super().__init__(A=A, n_concepts=n_concepts, n_features=n_features)

        # initialize autoencoder
        self.save_path = os.path.join(save_path, "weights")
        self.concept_space = concept_space

        if self.concept_space == "any":
            self.autoencoder = _AnyAutoEncoder(self.n_features, n_concepts, encoder_layers)
        elif self.concept_space == "binary":
            self.autoencoder = _PositiveBinaryAutoEncoder(self.n_features, n_concepts, encoder_layers)
        elif self.concept_space == "bounded":
            self.autoencoder = _BoundedAutoEncoder(self.n_features, n_concepts, encoder_layers)
        elif self.concept_space == "positive":
            self.autoencoder = _PositiveAutoEncoder(self.n_features, n_concepts, encoder_layers)
        elif self.concept_space == "positive_bounded":
            self.autoencoder = _PositiveBoundedAutoEncoder(self.n_features, n_concepts, encoder_layers)
        elif self.concept_space == "ternary":
            self.autoencoder = _TernaryAutoEncoder(self.n_features, n_concepts, encoder_layers)
        else:
            raise ValueError("concept_space must be one of 'positive', 'any', 'binary', 'bounded', 'positive_bounded', 'ternary'")
        
        if self.save_path is not None:
            if not force_recompute:
                try:
                    self.autoencoder.load_state_dict(torch.load(self.save_path))
                except FileNotFoundError:
                    force_recompute = True
        else:
            force_recompute = True
        
        if force_recompute:
            # training parameters
            self.epochs = (nb_steps * batch_size) // A.shape[0]
            self.normalize_decoder = normalize_decoder
            self.early_stopping = EarlyStopping(patience=int(early_stopping_patience * self.epochs))
            self.neurons_resampling = DeadNeuronsResampling(n_concepts=n_concepts, device=device)

            # loss
            self.reconstruction_loss = loss
            self.sparse_loss = lambda x: torch.norm(x, p=1, dim=1).mean()
            self.l1_coef = l1_coef
            self.loss = lambda inputs, concepts, outputs:\
                self.reconstruction_loss(inputs, outputs)  + self.l1_coef * self.sparse_loss(concepts)

            # setting seed to analyze stability
            torch.manual_seed(random_state)

            # training parameters
            assert hasattr(torch.optim, optimizer)
            self.optimizer = getattr(torch.optim, optimizer)(self.autoencoder.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=lr_scheduler_step_size,
                                                            gamma=lr_scheduler_gamma)

            # creating datasets
            train_loader, val_loader = self.split_and_create_dataloaders(A, batch_size, train_val_fraction)

            # training
            self.autoencoder.to(device)
            train_losses, val_losses = self.training_loop(train_loader, val_loader, device)
            self.autoencoder.eval()

            # load best model
            self.autoencoder.load_state_dict(torch.load(self.save_path))

            # training losses curves visualization
            if show_curves:
                plt.figure(figsize=(10, 7))
                plt.plot(train_losses, color='orange', label='train loss')
                plt.plot(val_losses, color='red', label='validataion loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.yscale("log")
                plt.legend()
                plt.grid()
                plt.show()
 
    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, x):
        return self.autoencoder.decode(x)
    
    def forward(self, x):
        return self.autoencoder(x)
    
    def to(self, device):
        self.autoencoder.to(device)
        return self

    @property
    def is_differentiable(self):
        return True
    
    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.autoencoder.decoder.weight.data.T
    
    def normalize_decoder_weights(self):
        with torch.no_grad():
            w = self.autoencoder.decoder.weight.data
            norm = w.norm(p=2, dim=1, keepdim=True).mean()
            self.autoencoder.decoder.weight.data = w.div(norm)

    def split_and_create_dataloaders(self, dataset, batch_size, train_val_fraction):
        generator = torch.Generator().manual_seed(0)
        nb_samples = len(dataset)
        splits = [int(train_val_fraction*nb_samples),
                  nb_samples - int(train_val_fraction*nb_samples)]
        trainset, valset = random_split(dataset, lengths=splits, generator=generator)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        return trainloader, valloader

    def training_epoch(self, train_loader, device, epoch):
        self.autoencoder.train()
        train_loss = 0.0
        for features in train_loader:
            # compute batch predictions
            features = features.to(device)
            self.optimizer.zero_grad()

            # make predictions
            concepts = self.encode(features)
            outputs = self.decode(concepts)

            # compute loss as the sum of the reconstruction error and the concepts sparsity
            loss = self.loss(features, concepts, outputs)

            # backpropagate loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            self.neurons_resampling.monitor(concepts)
            # resample dead neurons
            if self.neurons_resampling.should_resample and epoch / self.epochs < 0.8:
                self.neurons_resampling.resample(self.autoencoder, train_loader, device)

            # force decoder weights to have unit norm
            if self.normalize_decoder:
                self.normalize_decoder_weights()
        
        train_loss = train_loss / len(train_loader)
        return train_loss

    def validating_epoch(self, val_loader, device):
        self.autoencoder.eval()
        val_loss = 0.0
        reconstruction_loss = 0.0
        sparsity_loss = 0.0
        with torch.no_grad():
            for features in val_loader:
                # compute batch predictions
                features = features.to(device)
                concepts = self.encode(features)
                outputs = self.decode(concepts)

                # get reconstruction loss
                loss = self.loss(features, concepts, outputs)
                reconstruction_loss += self.reconstruction_loss(features, outputs)
                sparsity_loss += self.sparse_loss(concepts)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        # print(f"\tDEBUG: validating epoch: val loss decomposed: {val_loss:.4f}, "+\
        #       f"reconstruction: {reconstruction_loss / len(val_loader):.4f}, "+\
        #       f"sparse: {self.l1_coef * sparsity_loss / len(val_loader):.4f}")
        return val_loss

    def training_loop(self, train_loader, val_loader, device):
        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        for epoch in range(self.epochs):
            # concepts = self.autoencoder.encode(self.A)  # TODO remove
            # print(f"\tDEBUG: training_loop epoch {epoch}: min: {concepts.min():.4f}, mean: {concepts.mean():.4f}, max: {concepts.max():.4f}, l1: {self.sparse_loss(concepts):.4f}, norm: {concepts.norm(p=2, dim=1).mean():.4f}")

            train_epoch_loss = self.training_epoch(train_loader, device, epoch)
            train_losses.append(train_epoch_loss)

            val_epoch_loss = self.validating_epoch(val_loader, device)
            val_losses.append(val_epoch_loss)

            # if (epoch % (self.epochs // 20)) == (self.epochs // 20 - 1):
            #     print(f"\t\tEpoch: {epoch+1}/{self.epochs}, " + \
            #             f"train loss: {train_epoch_loss:.4f}, val loss: {val_epoch_loss:.4f}")  # TODO: remove
            self.scheduler.step()

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.autoencoder.state_dict(), self.save_path)

            self.early_stopping(val_epoch_loss)
            if self.early_stopping.early_stop:
                break

        return train_losses, val_losses
