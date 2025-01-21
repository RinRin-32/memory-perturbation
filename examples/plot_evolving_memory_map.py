import os
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Plot memory maps and decision boundaries')
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

        # Check if decision boundary data is available
        if 'decision_boundary' in scores_dict:
            xx, yy, Z = scores_dict['decision_boundary']['xx'], scores_dict['decision_boundary']['yy'], scores_dict['decision_boundary']['Z']
            X_train = scores_dict['X_train']
            y_train = scores_dict['y_train']

            # Plot decision boundary
            fig, ax = plt.subplots()
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
            scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k')
            plt.title(f"Decision Boundary at Epoch {epoch}", fontsize=fontsize)
            plt.xlabel('Feature 1', fontsize=fontsize)
            plt.ylabel('Feature 2', fontsize=fontsize)
            plt.colorbar(scatter, ax=ax, label='Class')
            plt.xticks(fontsize=fontsize - 2)
            plt.yticks(fontsize=fontsize - 2)

            # Save the decision boundary plot
            decision_boundary_filename = os.path.join(dir, f"{args.name_exp}_decision_boundary_epoch_{epoch}.pdf")
            plt.tight_layout()
            plt.savefig(decision_boundary_filename, format="pdf")
            plt.show()
            plt.close()

            print(f"Saved decision boundary plot for epoch {epoch} to {decision_boundary_filename}")

        # Plot Bayesian Prediction Error vs Bayesian Leverage Score
        fig, ax = plt.subplots()
        plt.ylabel('Bayesian Prediction Error', fontsize=fontsize, labelpad=15)
        plt.xlabel('Bayesian Leverage Score', fontsize=fontsize)
        plt.yticks(fontsize=fontsize - 2)
        plt.xticks(fontsize=fontsize - 2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.tick_params(axis='both', which='major', labelcolor='gray')
        plt.grid(True, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.scatter(bls, bpe, edgecolor='r', facecolor='w', marker=".", s=75)

        # Save the BPE vs. BLS plot
        memory_map_filename = os.path.join(dir, f"{args.name_exp}_evolving_memory_map_epoch_{epoch}.pdf")
        plt.tight_layout()
        plt.savefig(memory_map_filename, format="pdf")
        plt.show()
        plt.close()

        print(f"Saved memory map plot for epoch {epoch} to {memory_map_filename}")