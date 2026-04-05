# explainability.py
# implements grad-cam and score-cam for visualising what the model looks at
# both methods output a heatmap that can be overlaid on the original image

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# helper functions

def _normalise(cam: np.ndarray) -> np.ndarray:
    # clip negatives and scale to 0-1
    cam = np.maximum(cam, 0)
    if cam.max() > 1e-8:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _upsample(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # resize the feature map back up to input image size
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


# Grad-CAM

class GradCAM:
    # uses gradients to figure out which parts of the image matter for the prediction
    # hooks into the last conv layer and uses the gradient signal to weight activations

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations : Optional[torch.Tensor] = None
        self._gradients   : Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # save activations during forward pass
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # save gradients during backward pass
            self._gradients = grad_output[0].detach()

        self._fh = self.target_layer.register_forward_hook(forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        # clean up hooks when done
        self._fh.remove()
        self._bh.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        # returns a heatmap showing what the model focused on
        # class_idx is optional - defaults to top predicted class
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # forward pass
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # backward pass for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # global average pool the gradients to get per-channel weights
        gradients   = self._gradients[0]
        activations = self._activations[0]
        weights     = gradients.mean(dim=(1, 2))

        # weighted sum of activations then relu
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for k, w in enumerate(weights):
            cam += w * activations[k]
        cam = F.relu(cam)

        # upsample back to input size and normalise
        h, w     = input_tensor.shape[2:]
        cam_up   = _upsample(cam.unsqueeze(0).unsqueeze(0), (h, w))
        heatmap  = _normalise(cam_up.squeeze().cpu().numpy())
        return heatmap


# Score-CAM

class ScoreCAM:
    # like grad-cam but doesnt use gradients at all
    # instead it masks the input with each activation map and sees how much the score changes

    def __init__(
        self,
        model       : nn.Module,
        target_layer: nn.Module,
        batch_size  : int = 32,
    ):
        self.model        = model
        self.target_layer = target_layer
        self.batch_size   = batch_size  # lower this if you run out of memory
        self._activations : Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()
        self._fh = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self):
        self._fh.remove()

    @torch.no_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx   : Optional[int] = None,
        device      : Optional[torch.device] = None,
    ) -> np.ndarray:
        # returns a heatmap - slower than grad-cam but no gradients needed
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        input_tensor = input_tensor.to(device)
        h, w = input_tensor.shape[2:]

        # get the predicted class and capture activations
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        activations = self._activations[0]
        n_channels  = activations.shape[0]

        # baseline score using a blank image
        baseline    = torch.zeros_like(input_tensor)
        base_score  = torch.softmax(self.model(baseline), dim=1)[0, class_idx].item()

        # upsample each activation map and use it as a mask on the input
        masks = []
        for k in range(n_channels):
            ak = activations[k].unsqueeze(0).unsqueeze(0)
            mk = _upsample(ak, (h, w)).squeeze()
            mn, mx = mk.min(), mk.max()
            mk = (mk - mn) / (mx - mn + 1e-8)
            masks.append(mk)
        masks = torch.stack(masks, dim=0)

        # apply masks to the input image
        input_exp = input_tensor.expand(n_channels, -1, -1, -1).clone()
        for k in range(n_channels):
            input_exp[k] = input_exp[k] * masks[k].unsqueeze(0)

        # run forward passes in batches to avoid running out of memory
        scores = []
        for start in range(0, n_channels, self.batch_size):
            batch  = input_exp[start : start + self.batch_size].to(device)
            logits = self.model(batch)
            probs  = torch.softmax(logits, dim=1)[:, class_idx]
            scores.append(probs.cpu())
        scores = torch.cat(scores, dim=0)

        # weight each activation map by how much it changed the score
        weights = (scores - base_score)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for k in range(n_channels):
            cam += weights[k].item() * activations[k].cpu()
        cam = F.relu(cam)

        # upsample and normalise
        cam_up  = _upsample(cam.unsqueeze(0).unsqueeze(0), (h, w))
        heatmap = _normalise(cam_up.squeeze().numpy())
        return heatmap
