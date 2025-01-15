import os
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Plot memory maps')
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load sensitivities
    dir = 'pickles/'
    file = open(dir + args.name_exp + '_memory_maps_scores.pkl', 'rb')
    scores_dict = pickle.load(file)
    file.close()

    bpe = scores_dict['bpe']
    bls = scores_dict['bls']

    # Plot Bayesian Prediction Error vs Bayesian Leverage Score
    fig, ax = plt.subplots()

    fontsize = 23
    plt.ylabel('Bayesian Prediction Error', fontsize=fontsize, labelpad=15)
    plt.xlabel('Bayesian Leverage Score', fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tick_params(axis='both', which='major', labelcolor='gray')
    plt.grid(True, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.scatter(bls, bpe, edgecolor='r', facecolor='w', marker=".", s=75)

    # Save first figure
    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.tight_layout()
    plt.savefig(dir + args.name_exp + '_memory_map.pdf', format="pdf")
    plt.show()
    plt.close()