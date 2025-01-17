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

    # Load scores for multiple epochs
    dir = 'pickles/'
    file_path = os.path.join(dir, f"{args.name_exp}_evolving_memory_maps_scores.pkl")
    with open(file_path, 'rb') as file:
        all_scores = pickle.load(file)

    # Plotting settings
    fontsize = 23
    dir = 'plots/'
    os.makedirs(dir, exist_ok=True)

    for epoch, scores_dict in all_scores.items():
        bpe = scores_dict['bpe']
        bls = scores_dict['bls']

        # Plot Bayesian Prediction Error vs Bayesian Leverage Score for each epoch
        fig, ax = plt.subplots()
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

        # Save the figure for the current epoch
        plot_filename = os.path.join(dir, f"{args.name_exp}_evolving_memory_map_epoch_{epoch}.pdf")
        plt.tight_layout()
        plt.savefig(plot_filename, format="pdf")
        plt.show()
        plt.close()

        print(f"Saved plot for epoch {epoch} to {plot_filename}")