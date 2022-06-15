import torch
from numpy import random

class Attack():
    # FGSM attack
    def fgsm_attack(self, X, y, model, loss_fn, epsilon):
        delta = torch.zeros_like(X, requires_grad=True)
        loss = loss_fn(model(X + delta), y)
        loss.backward()
        # Collect the element-wise sign of the data gradient
        sign_data_grad = delta.grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        X = X + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        X = torch.clamp(X, 0, 1)
        # Return the perturbed image
        return X
