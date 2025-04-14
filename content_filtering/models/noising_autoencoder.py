import torch
import torch.nn as nn
from setup import *


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim,
                 encoder_layers,
                 latent_dim,
                 decoder_layers,
                 noise_std):
        """
        builds a denoising autoencoder
        :param input_dim: dim of input
        :param encoder_layers: dims of encoding layers
        :param latent_dim: dim of bottleneck layer
        :param decoder_layers: dims of decoding layers
        :param noise_std: standard deviation of Gaussian noise added during training
        """
        super(DenoisingAutoencoder, self).__init__()
        self.noise_std = noise_std

        # Build the encoder dynamically
        encoder_modules = []
        prev_dim = input_dim
        for layer_dim in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, layer_dim))
            encoder_modules.append(nn.ReLU())
            prev_dim = layer_dim
        # Final encoder layer to latent space
        encoder_modules.append(nn.Linear(prev_dim, latent_dim))
        encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        # Build the decoder dynamically
        decoder_modules = []
        prev_dim = latent_dim
        for layer_dim in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, layer_dim))
            decoder_modules.append(nn.ReLU())
            prev_dim = layer_dim
        # Final decoder layer mapping back to input dimension
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        # Using sigmoid to constrain the outputs in [0, 1]
        decoder_modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        """
        forward pass of the model
        :param x: input features
        :return: embeddings and reconstructed input
        """
        # Add noise during training
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        # Encode with noise.
        latent = self.encoder(x_noisy)
        # Decode from the latent vector
        output = self.decoder(latent)
        return output, latent

    def encode(self, x):
        """
        encodes the input into embedding space
        :param x: input
        :return: embeddings for x
        """
        self.eval()
        with torch.no_grad():
            latent = self.encoder(x)
        return latent


class MSE_BCE_Loss(nn.Module):
    def __init__(self, split_index, mse_weight=1, bce_weight=.5):
        """
        custom loss function that combines MSE for continuous features and BCE for binary ones and scales them to
        account for scale differences
        :param split_index: idx where binary values start
        :param mse_weight: weight for mse error
        :param bce_weight: weight for bce error
        """
        super(MSE_BCE_Loss, self).__init__()
        self.split_index = split_index
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight

    def forward(self, reconstructed, original):
        """
        calculates the custom loss
        :param reconstructed: output of the autoencoder
        :param original: input before noise was added
        :return: custom loss result
        """
        # Continuous part loss calculation.
        continuous_recon = reconstructed[:, :self.split_index]
        continuous_orig = original[:, :self.split_index]
        mse = self.mse_loss(continuous_recon, continuous_orig)

        # Binary part loss calculation.
        binary_recon = reconstructed[:, self.split_index:]
        binary_orig = original[:, self.split_index:]
        bce = self.bce_loss(binary_recon, binary_orig)

        # Combine the losses; you can add weights to each term if needed.
        return mse * self.mse_weight + bce * self.bce_weight


def build_autoencoder(input_dim,
                      encoder_layers=MODEL_PARAMS['encoder_layers'],
                      latent_dim=MODEL_PARAMS['latent_dim'],
                      decoder_layers=MODEL_PARAMS['decoder_layers'],
                      noise_std=MODEL_PARAMS['noise_std']):
    """
    helper function to instantiate the autoencoder with the given parameters
    :param input_dim: dim of input
    :param encoder_layers: dims of encoding layers
    :param latent_dim: dim of bottleneck layer
    :param decoder_layers: dims of decoding layers
    :param noise_std: standard deviation of Gaussian noise added during training
    :return: autoencoder model
    """
    model = DenoisingAutoencoder(input_dim, encoder_layers, latent_dim, decoder_layers, noise_std)
    return model
