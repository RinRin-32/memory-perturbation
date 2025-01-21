import os
import pickle
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Compare sensitivities to true deviations in softmax outputs from retraining.')
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load sensitivities
    dir = 'pickles/'
    with open(dir + args.name_exp + '_evolving_memory_maps_scores.pkl', 'rb') as file:
        all_scores = pickle.load(file)
    file.close()

    # Load results from retraining
    with open(dir + args.name_exp + '_evolving_memory_maps_retrain.pkl', 'rb') as file:
        retrain = pickle.load(file)
    file.close()

    for epoch, ((scores_key, scores_dict), (retrain_key, retrain_dict)) in enumerate(zip(all_scores.items(), retrain.items())):
        sensitivities = scores_dict['sensitivities']
        indices_retrain = retrain_dict['indices_retrain']
        softmax_deviations = retrain_dict['softmax_deviations']

        # Figure
        fig, ax = plt.subplots()

        fontsize = 23
        plt.ylabel('Estimated Deviation', fontsize=fontsize, labelpad=15)
        plt.xlabel('True Deviation', fontsize=fontsize)
        plt.yticks(fontsize=fontsize-2)
        plt.xticks(fontsize=fontsize-2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.tick_params(axis='both', which='major', labelcolor='gray')
        plt.grid(True, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Plot results
        xpoints = ypoints = plt.ylim()
        plt.plot(xpoints, ypoints, linestyle='--', color='0.5', lw=5, scalex=False, scaley=False, zorder=-10)
        plt.scatter(softmax_deviations, sensitivities[indices_retrain], edgecolor='r', facecolor='w', marker="o", s=75)

        # Save figure
        dir = 'plots/'
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'{dir}{args.name_exp}_{epoch}_validation.pdf', format="pdf")
        plt.show()
        plt.close()