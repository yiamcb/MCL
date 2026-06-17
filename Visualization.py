"""
Visualization.py
================
Plots built from REAL model outputs: t-SNE of fused embeddings, confusion
matrices, and per-subject accuracy distributions. Use these to produce the
cross-subject figures from measured results.
"""
import numpy as np
import matplotlib.pyplot as plt


def extract_embeddings(model, X):
    """Activations of the last hidden layer (the fused embedding) for input X."""
    import tensorflow as tf
    sub = tf.keras.Model(model.inputs, model.layers[-2].output)
    return sub.predict(X, verbose=0)


def plot_tsne(embeddings, labels, class_names=None, ax=None, perplexity=30, seed=0):
    from sklearn.manifold import TSNE
    Z = TSNE(n_components=2, perplexity=perplexity, init="pca",
             random_state=seed).fit_transform(embeddings)
    ax = ax or plt.gca()
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, edgecolor="none",
                   label=(class_names[c] if class_names else str(c)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8)
    return ax


def plot_confusion(y_true, y_pred, class_names=None, ax=None, cmap="Blues"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax = ax or plt.gca(); ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    k = cm.shape[0]; names = class_names or list(range(k))
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]*100:.1f}%", ha="center",
                    va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    return ax


def plot_per_subject(per_subject_scores, ax=None):
    ax = ax or plt.gca()
    names = list(per_subject_scores.keys())
    data = [np.asarray(per_subject_scores[n], float) for n in names]
    ax.violinplot(data, showmeans=False, showextrema=False)
    ax.boxplot(data, widths=0.25, showfliers=False)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_ylabel("Per-subject accuracy (%)")
    return ax
