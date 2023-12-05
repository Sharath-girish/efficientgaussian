import torch
import numpy as np

class EntropyLoss:
    def __init__(self, prob_models, lambdas, noise_freq=1):
        self.prob_models = prob_models
        self.lambdas = list(lambdas.values())
        self.noise_freq = noise_freq
        self.start_idx, self.end_idx = [0], []
        for prob_model in prob_models.values():
            self.start_idx += [self.start_idx[-1] + prob_model.num_channels]
            self.end_idx += [self.start_idx[-1]]
        self.net_channels = self.start_idx[-1]
        self.noise = None

    def loss(self, latents, iteration, is_val=False):
        latents = [l for param_name,l in latents.items() if param_name in self.prob_models]
        if len(latents) == 0:
            return 0.0, 0.0
        noise = self.noise
        if not is_val:
            if self.noise_freq == 1:
                noise = torch.rand(latents[0].shape[0],self.net_channels).to(latents[0])-0.5
            elif (iteration-1) % self.noise_freq == 0:
                self.noise = torch.rand(latents[0].shape[0],self.net_channels).to(latents[0])-0.5
                noise = self.noise
        total_bits, total_loss = 0.0, 0.0
        for i,prob_model in enumerate(self.prob_models.values()):
            weight = latents[i] + noise[:,self.start_idx[i]:self.end_idx[i]] if not is_val else torch.round(latents[i])
            weight_p, weight_n = weight + 0.5, weight - 0.5
            prob = prob_model(weight_p) - prob_model(weight_n)
            bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
            total_bits += bits
            total_loss += self.lambdas[i] * bits
        return total_loss / latents[0].shape[0], total_bits / latents[0].shape[0]
