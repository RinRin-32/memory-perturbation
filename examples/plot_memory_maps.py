import os
import pickle
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Plot memory maps')
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load sensitivities
    dir = 'pickles/'
    file = open(dir + args.name_exp + '_memory_maps.pkl', 'rb')
    scores_dict = pickle.load(file)
    file.close()
    bpe = scores_dict['bpe']
    bls = scores_dict['bls']

    # Figure
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

    # Plot results
    #xpoints = ypoints = plt.ylim()
    #plt.plot(xpoints, ypoints, linestyle='--', color='0.5', lw=5, scalex=False, scaley=False, zorder=-10)
    plt.scatter(bls, bpe, edgecolor='r', facecolor='w', marker=".", s=75)

    # Save figure
    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.tight_layout()
    plt.savefig(dir+args.name_exp+'_memory_map.pdf', format="pdf")
    plt.show()
    plt.close()







