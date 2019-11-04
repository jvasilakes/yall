import numpy as np
import matplotlib.pyplot as plt


# TODO: Reference the active learning challenge
def compute_alc(aucs, normalize=True):
    '''
    Compute the normalized Area under the Learning Curve (ALC)
    for a set of AUCs.

    param aucs: np.array of AUC values.
    param normalize: Whether to normalize the ALC, default: True.
    '''
    alc = np.trapz(aucs, dx=1.0)
    if normalize is True:
        random_auc = np.repeat(0.5, aucs.shape[0])
        A_random = np.trapz(random_auc, dx=1.0)
        max_auc = np.repeat(1.0, aucs.shape[0])
        A_max = np.trapz(max_auc, dx=1.0)
        # Normalize the ALC
        alc = (alc - A_random) / (A_max - A_random)
    return alc


def plot_learning_curve(aucs, L_init, L_end,
                        title="ALC", eval_metric="auc", saveto=None):
    '''
    Plots the learning curve for a set of AUCs.

    param aucs: np.array of AUC values.
    param L_init: The initial size of the labeled set.
    param L_end: The final size of the labeled set.
    param title: The title of this plot.
    param saveto: Filename to which to save this plot instead of showing it.
    '''
    alc = compute_alc(aucs)
    draws = np.arange(L_init, L_end)
    plt.figure(figsize=(10, 5))
    plt.plot(draws, aucs, linewidth=2)
    plt.xlabel("|L|", fontsize=15)
    plt.ylabel(eval_metric.upper(), fontsize=15)
    plt.title(title)
    plt.annotate(f"ALC: {alc:.4f}", xy=(0.8, 0.05), xycoords="axes fraction")
    if saveto is not None:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()
