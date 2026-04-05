# demo.py
# runs grad-cam and score-cam on the stanford dogs dataset
# can do quantitative evaluation, visualisation, or both
#
# usage:
#   python demo.py --data_dir /path/to/stanford_dogs --mode both
#   if no model file is found it will train one first

import argparse
import os
import time
import csv

import matplotlib
matplotlib.use("Agg")            # headless backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model          import build_model, train_model, load_model, get_dataloaders, DEVICE
from explainability import GradCAM, ScoreCAM
from evaluation     import evaluate_method


# transform for inference (no augmentation)
infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# inverse normalisation so we can display the image
denorm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std =[1/0.229,       1/0.224,       1/0.225],
)


# visualisation helpers

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    # converts a normalised tensor back to a displayable rgb image
    img = denorm(tensor.squeeze(0).cpu()).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    # blends the heatmap onto the image using jet colormap
    coloured = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended  = (alpha * coloured + (1 - alpha) * rgb).astype(np.uint8)
    return blended


def save_comparison_figure(
    rgb        : np.ndarray,
    grad_cam   : np.ndarray,
    score_cam  : np.ndarray,
    true_label : str,
    pred_label : str,
    save_path  : str,
):
    # saves a side by side figure: original, grad-cam, score-cam
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"True: {true_label}   |   Predicted: {pred_label}",
        fontsize=11, fontweight="bold", y=1.01
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap(rgb, grad_cam))
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap(rgb, score_cam))
    axes[2].set_title("Score-CAM", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# breed pairs that are commonly confused - these are interesting to visualise
INTERESTING_PAIRS = [
    ("n02110185-Siberian_husky",        "n02110063-malamute"),
    ("n02106166-Border_collie",         "n02106550-Rottweiler"),
    ("n02099601-golden_retriever",      "n02099712-Labrador_retriever"),
    ("n02085620-Chihuahua",             "n02086240-Shih-Tzu"),
    ("n02096294-Australian_terrier",    "n02096177-cairn"),
]


def collect_misclassified(model, data_dir, class_names, max_per_pair=4):
    # finds misclassified images for the breed pairs above
    # returns a list of dicts with the image tensor and label info
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    dataset = ImageFolder(
        root=os.path.join(data_dir, "Images"),
        transform=infer_transform,
    )
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # figure out which class indices we care about
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    target_pairs = set()
    for a, b in INTERESTING_PAIRS:
        if a in name_to_idx and b in name_to_idx:
            target_pairs.add((name_to_idx[a], name_to_idx[b]))
            target_pairs.add((name_to_idx[b], name_to_idx[a]))

    collected   = []
    seen_pairs  = {}

    model.eval()
    with torch.no_grad():
        for img, label in loader:
            img      = img.to(DEVICE)
            true_idx = label.item()
            pred_idx = model(img).argmax(1).item()

            if pred_idx == true_idx:
                continue
            if (true_idx, pred_idx) not in target_pairs:
                continue

            pair_key = tuple(sorted([true_idx, pred_idx]))
            if seen_pairs.get(pair_key, 0) >= max_per_pair:
                continue

            seen_pairs[pair_key] = seen_pairs.get(pair_key, 0) + 1
            collected.append({
                "img_tensor": img.cpu(),
                "true_idx"  : true_idx,
                "pred_idx"  : pred_idx,
                "true_name" : class_names[true_idx].split("-", 1)[-1].replace("_", " "),
                "pred_name" : class_names[pred_idx].split("-", 1)[-1].replace("_", " "),
            })

            total_needed = len(target_pairs) // 2 * max_per_pair
            if len(collected) >= total_needed:
                break

    print(f"Collected {len(collected)} misclassified samples from interesting pairs.")
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root of Stanford Dogs Dataset.")
    parser.add_argument("--model_path", type=str, default="resnet50_dogs.pth",
                        help="Path to save/load the fine-tuned model.")
    parser.add_argument("--out_dir",    type=str, default="outputs",
                        help="Directory for figures and CSV results.")
    parser.add_argument("--mode",       type=str, default="both",
                        choices=["eval", "visualise", "both"])
    parser.add_argument("--n_eval",     type=int, default=200,
                        help="Number of images for quantitative evaluation.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load or train the model
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir)

    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path} …")
        model = load_model(args.model_path)
    else:
        print("No saved model found. Training from scratch …")
        model = build_model()
        model = train_model(model, train_loader, val_loader,
                            save_path=args.model_path)

    model.eval()
    target_layer = model.layer4[-1]      # last resnet block

    # set up both explanation methods
    grad_cam  = GradCAM(model, target_layer)
    score_cam = ScoreCAM(model, target_layer, batch_size=32)

    # quantitative evaluation
    if args.mode in ("eval", "both"):
        print("\n── Quantitative Evaluation ─────────────────────────────")

        from torchvision.datasets import ImageFolder
        from torch.utils.data    import DataLoader
        eval_dataset = ImageFolder(
            root=os.path.join(args.data_dir, "Images"),
            transform=infer_transform,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=1,
                                 shuffle=False, num_workers=2)

        print("Evaluating Grad-CAM …")
        t0 = time.time()
        grad_results = evaluate_method(
            model, lambda img, cls: grad_cam(img, cls),
            eval_loader, n_images=args.n_eval, device=DEVICE
        )
        grad_time = (time.time() - t0) / args.n_eval

        print("Evaluating Score-CAM …")
        t0 = time.time()
        score_results = evaluate_method(
            model, lambda img, cls: score_cam(img, cls),
            eval_loader, n_images=args.n_eval, device=DEVICE
        )
        score_time = (time.time() - t0) / args.n_eval

        # print results
        print("\nResults:")
        print(f"  Insertion AUC  (higher is better):  Grad-CAM = {grad_results['insertion_auc']:.4f}   Score-CAM = {score_results['insertion_auc']:.4f}")
        print(f"  Deletion AUC   (lower is better):   Grad-CAM = {grad_results['deletion_auc']:.4f}   Score-CAM = {score_results['deletion_auc']:.4f}")
        print(f"  Avg time/image (seconds):           Grad-CAM = {grad_time:.4f}   Score-CAM = {score_time:.4f}")

        # save results to csv
        csv_path = os.path.join(args.out_dir, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "grad_cam", "score_cam"])
            writer.writerow(["insertion_auc",
                             grad_results["insertion_auc"],
                             score_results["insertion_auc"]])
            writer.writerow(["deletion_auc",
                             grad_results["deletion_auc"],
                             score_results["deletion_auc"]])
            writer.writerow(["avg_time_seconds", grad_time, score_time])
        print(f"Results saved to {csv_path}")

    # qualitative visualisation
    if args.mode in ("visualise", "both"):
        print("\n── Qualitative Visualisation ───────────────────────────")
        samples = collect_misclassified(model, args.data_dir, class_names)

        for i, s in enumerate(samples):
            img    = s["img_tensor"].to(DEVICE)
            rgb    = tensor_to_rgb(s["img_tensor"])
            true_n = s["true_name"]
            pred_n = s["pred_name"]

            gc_map = grad_cam(img,  class_idx=s["pred_idx"])
            sc_map = score_cam(img, class_idx=s["pred_idx"])

            fname = os.path.join(args.out_dir,
                                 f"sample_{i:02d}_{true_n.replace(' ','_')}"
                                 f"_vs_{pred_n.replace(' ','_')}.png")
            save_comparison_figure(rgb, gc_map, sc_map, true_n, pred_n, fname)
            print(f"  Saved: {fname}")

    grad_cam.remove_hooks()
    score_cam.remove_hooks()
    print("\nDone.")


if __name__ == "__main__":
    main()
