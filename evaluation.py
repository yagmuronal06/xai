# evaluation.py
# metrics to measure how good the saliency maps are
# uses deletion auc, insertion auc, and pointing game accuracy

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable


# deletion and insertion auc

def _perturb_auc(
    model       : torch.nn.Module,
    input_tensor: torch.Tensor,
    heatmap     : np.ndarray,
    class_idx   : int,
    n_steps     : int = 50,
    mode        : str = "deletion",   # "deletion" or "insertion"
    device      : torch.device = torch.device("cpu"),
) -> float:
    # computes deletion or insertion auc for one image
    # deletion = remove most important pixels and see how fast confidence drops
    # insertion = reveal most important pixels and see how fast confidence rises
    model.eval()
    input_tensor = input_tensor.to(device)
    h, w         = input_tensor.shape[2:]
    n_pixels     = h * w

    # sort pixels by importance, most important first
    flat_importance = heatmap.flatten()
    sorted_idx      = np.argsort(flat_importance)[::-1]

    scores = []
    for step in range(n_steps + 1):
        n_perturbed = int(step / n_steps * n_pixels)
        perturbed   = input_tensor.clone()

        if mode == "deletion":
            # zero out the top-k pixels
            mask = torch.ones(h * w, device=device)
            if n_perturbed > 0:
                mask[torch.from_numpy(sorted_idx[:n_perturbed].copy()).to(device)] = 0.0
            mask = mask.view(1, 1, h, w).expand_as(perturbed)
            perturbed = perturbed * mask

        else:  # insertion
            # start from blurred image and reveal top-k pixels
            baseline = F.avg_pool2d(input_tensor, kernel_size=11, stride=1, padding=5)
            mask     = torch.zeros(h * w, device=device)
            if n_perturbed > 0:
                mask[torch.from_numpy(sorted_idx[:n_perturbed].copy()).to(device)] = 1.0
            mask     = mask.view(1, 1, h, w).expand_as(perturbed)
            perturbed = baseline * (1 - mask) + input_tensor * mask

        with torch.no_grad():
            logits = model(perturbed)
            prob   = torch.softmax(logits, dim=1)[0, class_idx].item()
        scores.append(prob)

    # area under the curve using trapezoid rule
    auc = float(np.trapz(scores, dx=1.0 / n_steps))
    return auc


def deletion_auc(model, input_tensor, heatmap, class_idx,
                 n_steps=50, device=torch.device("cpu")) -> float:
    # lower is better
    return _perturb_auc(model, input_tensor, heatmap, class_idx,
                        n_steps=n_steps, mode="deletion", device=device)


def insertion_auc(model, input_tensor, heatmap, class_idx,
                  n_steps=50, device=torch.device("cpu")) -> float:
    # higher is better
    return _perturb_auc(model, input_tensor, heatmap, class_idx,
                        n_steps=n_steps, mode="insertion", device=device)


# pointing game

def pointing_game_hit(
    heatmap    : np.ndarray,
    bbox       : tuple[int, int, int, int],
) -> bool:
    # checks if the peak of the heatmap is inside the bounding box
    # bbox format is (x_min, y_min, x_max, y_max)
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    x_min, y_min, x_max, y_max = bbox
    return (x_min <= peak_x <= x_max) and (y_min <= peak_y <= y_max)


def pointing_game_accuracy(
    heatmaps : list[np.ndarray],
    bboxes   : list[tuple[int, int, int, int]],
) -> float:
    # fraction of images where the peak activation landed inside the bounding box
    assert len(heatmaps) == len(bboxes), "Mismatch between heatmaps and bboxes."
    hits = sum(pointing_game_hit(h, b) for h, b in zip(heatmaps, bboxes))
    return hits / len(heatmaps)


# runs all metrics over a dataset

def evaluate_method(
    model          : torch.nn.Module,
    explain_fn     : Callable,
    dataloader     : torch.utils.data.DataLoader,
    bboxes         : list[tuple] | None = None,
    n_images       : int = 200,
    n_steps        : int = 50,
    device         : torch.device = torch.device("cpu"),
    verbose        : bool = True,
) -> dict:
    # loops over images and computes deletion/insertion auc for each
    # also does pointing game if bboxes are provided
    # returns a dict with the averaged results
    model.eval()
    del_aucs, ins_aucs = [], []
    heatmaps_for_pg    = []
    bboxes_for_pg      = []

    for i, (img, label) in enumerate(dataloader):
        if i >= n_images:
            break

        img      = img.to(device)
        class_idx = label.item()

        heatmap  = explain_fn(img, class_idx)

        del_aucs.append(deletion_auc(model, img, heatmap, class_idx,
                                     n_steps=n_steps, device=device))
        ins_aucs.append(insertion_auc(model, img, heatmap, class_idx,
                                      n_steps=n_steps, device=device))

        if bboxes is not None:
            heatmaps_for_pg.append(heatmap)
            bboxes_for_pg.append(bboxes[i])

        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_images}]  "
                  f"Del AUC: {np.mean(del_aucs):.4f}  "
                  f"Ins AUC: {np.mean(ins_aucs):.4f}")

    results = {
        "deletion_auc"  : float(np.mean(del_aucs)),
        "insertion_auc" : float(np.mean(ins_aucs)),
    }
    if bboxes is not None:
        results["pointing_game_acc"] = pointing_game_accuracy(
            heatmaps_for_pg, bboxes_for_pg
        )
    return results
