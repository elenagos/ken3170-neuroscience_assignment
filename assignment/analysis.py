import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import config

from core import (
    SantoroDataset,
    MODEL_CLASSES,
    extract_activations,
    load_yamnet_activations,
)
#Toolbox
def compute_rdm(patterns: np.ndarray, metric="euclidean") -> np.ndarray:
    n_classes = patterns.shape[0]
    rdm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(i + 1, n_classes):

            if metric == "euclidean":
                distance = np.linalg.norm(
                    patterns[i] - patterns[j]
                )

            elif metric == "correlation":
                r = np.corrcoef(
                    patterns[i],
                    patterns[j]
                )[0, 1]

                distance = 1 - r

            else:
                raise ValueError(
                    "metric must be 'euclidean' or 'correlation'"
                )

            rdm[i, j] = distance
            rdm[j, i] = distance

    return rdm
def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(rdm.shape[0], k=1)
    return rdm[indices]
def compare_rdms(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    method="pearson"
) -> float:

    if rdm_a.shape != rdm_b.shape:
        raise ValueError(
            f"RDM shapes differ: {rdm_a.shape} vs {rdm_b.shape}"
        )

    vec_a = upper_triangle(rdm_a)
    vec_b = upper_triangle(rdm_b)

    if method == "pearson":
        r, _ = pearsonr(vec_a, vec_b)

    elif method == "spearman":
        r, _ = spearmanr(vec_a, vec_b)

    else:
        raise ValueError(
            "method must be 'pearson' or 'spearman'"
        )

    return r
def flatten_activations(activations):
    activations = np.asarray(activations)
    return activations.reshape(activations.shape[0], -1)
def analyze_model(
    model_name,
    training_status,
    model,
    dataloader,
    brain_rdm,
    run=None,
    return_rdms=False,
):
    activations = extract_activations(dataloader, model)

    results = []
    rdms = {}

    for layer_index, (layer_name, acts) in enumerate(activations.items()):
        acts = acts.numpy()
        acts = flatten_activations(acts)

        layer_rdm = compute_rdm(
            acts,
            metric="euclidean"
        )

        rdms[layer_name] = layer_rdm
        score = compare_rdms(
            layer_rdm,
            brain_rdm,
            method="pearson"
        )

        results.append({
            "model": model_name,
            "training": training_status,
            "layer": layer_name,
            "run": run,
            "layer_index": layer_index,
            "n_layers": len(activations),
            "relative_depth": layer_index / max(len(activations) - 1, 1),
            "rsa": score
        })

        # Uncomment this when you want to see every layer score while it runs.
        # print(
        #     f"{model_name} | {training_status} | "
        #     f"{layer_name}: RSA = {score:.3f}"
        # )

    return (results, rdms) if return_rdms else results
# Load saved trained weights (no training).
def load_trained_model(model_name, checkpoint_path):
    model = MODEL_CLASSES[model_name](
        num_classes=config.NUM_CLASSES
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu", weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    #Loading dataset
    dataset = SantoroDataset()
    print("Number of sounds:", len(dataset))
    print("Brain responses shape:", dataset.brain_responses.shape)

    sample = dataset[0]

    print("Sample contains:")
    for item in sample:
        if hasattr(item, "shape"):
            print(type(item), item.shape)
        else:
            print(type(item), item)

    brain_activations = dataset.brain_responses.numpy()

    print("Brain activations shape:", brain_activations.shape)
    brain_rdm = compute_rdm(brain_activations)
    print("Brain RDM shape:", brain_rdm.shape)
    print("Symmetric:", np.allclose(brain_rdm, brain_rdm.T))
    print("Zero diagonal:", np.allclose(np.diag(brain_rdm), 0))
    #Visualisation
    plt.figure(figsize=(7, 6))

    plt.imshow(brain_rdm)
    plt.colorbar(label="Euclidean distance")

    plt.title("STG Brain RDM")
    plt.xlabel("Sound")
    plt.ylabel("Sound")

    plt.tight_layout()
    plt.show()
    #Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )
    #first untrained model
    waveform_model = MODEL_CLASSES["waveform"](
        num_classes=50
    )

    print(waveform_model)
    waveform_activations = extract_activations(
        dataloader,
        waveform_model
    )
    for layer_name, acts in waveform_activations.items():
        print(layer_name, acts.shape)
    #Count RSA for each layer
    for layer_name, acts in waveform_activations.items():

        acts = acts.numpy()
        acts = flatten_activations(acts)

        layer_rdm = compute_rdm(
            acts,
            metric="euclidean"
        )

        score = compare_rdms(
            layer_rdm,
            brain_rdm,
            method="pearson"
        )

        print(
            f"{layer_name}: RSA = {score:.3f}"
        )
    #Waveform model
    waveform_model = MODEL_CLASSES["waveform"](num_classes=50)

    waveform_results = analyze_model(
        "waveform",
        "untrained",
        waveform_model,
        dataloader,
        brain_rdm
    )
    #Uninspired model
    uninspired_model = MODEL_CLASSES["uninspired"](num_classes=50)

    uninspired_results = analyze_model(
        "uninspired",
        "untrained",
        uninspired_model,
        dataloader,
        brain_rdm
    )
    #Inspired model
    inspired_model = MODEL_CLASSES["inspired"](num_classes=50)

    inspired_results = analyze_model(
        "inspired",
        "untrained",
        inspired_model,
        dataloader,
        brain_rdm
    )
    #Combined results table
    all_untrained_results = (
        waveform_results
        + uninspired_results
        + inspired_results
    )
    untrained_df = pd.DataFrame(all_untrained_results)
    print(untrained_df)

    MODEL_DIR = Path(__file__).resolve().parent / "models"

    print("Model directory:", MODEL_DIR)
    print("Checkpoints:")

    for path in MODEL_DIR.glob("*.pt"):
        print(path.name)

    trained_models = {}
    trained_activations = {}
    trained_rdms = {}

    model_names = [
        "waveform",
        "uninspired",
        "inspired"
    ]

    for model_name in model_names:

        checkpoint = next(
            MODEL_DIR.glob(
                f"{model_name}_run*_best.pt"
            )
        )

        print(
            f"\nLoading {model_name}: "
            f"{checkpoint.name}"
        )

        # Build architecture + load trained weights
        model = load_trained_model(
            model_name,
            checkpoint
        )

        trained_models[model_name] = model

        # Extract layer activations
        activations = extract_activations(
            dataloader,
            model
        )

        trained_activations[model_name] = activations

        # Construct an RDM for every layer
        model_rdms = {}

        for layer_name, acts in activations.items():

            acts = acts.numpy()
            acts = flatten_activations(acts)

            rdm = compute_rdm(
                acts,
                metric="euclidean"
            )

            model_rdms[layer_name] = rdm

            print(
                f"{layer_name}: "
                f"activations {acts.shape}, "
                f"RDM {rdm.shape}"
            )

        trained_rdms[model_name] = model_rdms
        trained_results = []

        for model_name, model_rdms in trained_rdms.items():

            for layer_name, layer_rdm in model_rdms.items():
                score = compare_rdms(
                    layer_rdm,
                    brain_rdm,
                    method="pearson"
                )

                trained_results.append({
                    "model": model_name,
                    "training": "trained",
                    "layer": layer_name,
                    "rsa": score
                })

                print(
                    f"{model_name} | {layer_name}: "
                    f"RSA = {score:.3f}"
                )
    trained_df = pd.DataFrame(trained_results)

    print(trained_df)
    all_results_df = pd.concat(
        [
            untrained_df,
            trained_df
        ],
        ignore_index=True
    )

    print(all_results_df)
    #Yamnet embeddings
    yamnet_activations = load_yamnet_activations()

    yamnet_results = []

    for layer_name, acts in yamnet_activations.items():
        acts = flatten_activations(acts)

        layer_rdm = compute_rdm(
            acts,
            metric="euclidean"
        )

        score = compare_rdms(
            layer_rdm,
            brain_rdm,
            method="pearson"
        )

        yamnet_results.append({
            "model": "yamnet",
            "training": "pretrained",
            "layer": layer_name,
            "rsa": score
        })

        print(f"YAMNet | {layer_name}: RSA = {score:.3f}")
    yamnet_df = pd.DataFrame(yamnet_results)
    print(yamnet_df)
    all_results_df = pd.concat(
        [untrained_df, trained_df, yamnet_df],
        ignore_index=True
    )

if __name__ == "__main__":
    main()
