import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path
sns.set()
# matplotlib inline

def get_dfbox(metrics, path):
    # df_melodia = pd.DataFrame.from_csv("../outputs/Melodia_scores.csv")
    # df_bosch = pd.DataFrame.from_csv("../outputs/juanjo_mdb_scores.csv")
    # df_cnn = pd.DataFrame.from_csv("~/Code/ismir2017-deepsalience/outputs/CNNmel2_argmax_scores.csv")
    df_piero = pd.DataFrame.from_csv(os.path.join(path, "all_scores.csv"))
    boxdata = []
    for metric in metrics:
        boxdata.extend([
            df_piero[metric]
            # df_melodia[metric],
            # df_bosch[metric],
            # df_cnn[metric]
        ])

    dfbox = pd.DataFrame(np.array(boxdata).T)
    return dfbox

def plot(path):

    fig = plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    metrics = ['Voicing False Alarm', 'Voicing Recall', 'Raw Chroma Accuracy', 'Raw Pitch Accuracy', 'Overall Accuracy']
    data_df = get_dfbox(metrics, path)

    n_algs = 1
    n_metrics = len(metrics)
    positions = []
    k = 1
    for i in range(n_metrics):
        for j in range(n_algs):
            positions.append(k)
            k = k + 1
        k = k + 1

    # current_palette = ["#E1D89F", "#8EA8BD", "#CF6766"]
    current_palette = ["#E1D89F"]
    colors = current_palette*n_metrics
    box = plt.boxplot(
        data_df.values, widths=0.8, positions=positions,
        patch_artist=True, showmeans=True,
        medianprops={'color': 'k'},
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='k'),
        vert=False
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    show_yaxis = True
    xlim = [0, 1]
    legend_loc = 2

    plt.xlabel('Score')
    # if show_yaxis:
    #     plt.yticks(np.arange(2, 4*(n_metrics + 1) - 2, 4), metrics, rotation='horizontal', weight='bold')
    # else:
    #     plt.yticks(np.arange(2, 4*(n_metrics + 1) - 2, 4), ['']*len(metrics), rotation='horizontal')

    if show_yaxis:
        plt.yticks(np.arange(1, 2*n_metrics, 2), metrics, rotation='horizontal', weight='bold')
    else:
        plt.yticks(np.arange(0, n_metrics, 1), ['']*len(metrics), rotation='horizontal')

    # plt.plot([0, 1], [4, 4], '--', color='k')

    if xlim is not None:
        plt.xlim(xlim)

    if legend_loc is not None:
        # draw temporary red and blue lines and use them to create a legend
        # h_benetos, = plt.plot([1,1],'s',color=colors[0], markersize=10)
        # h_duan, = plt.plot([1,1],'s',color=colors[1], markersize=10)
        # h_cnn, = plt.plot([1,1],'s',color=colors[2], markersize=10)
        h_piero, = plt.plot([1,1],'s',color=colors[0], markersize=10)
        # lgd = plt.legend((h_cnn, h_duan, h_benetos),('CNN', 'Bosch', 'Salamon'), ncol=1, loc=legend_loc)
        lgd = plt.legend(([h_piero]), (['CNN-RNN']), ncol=1, loc=legend_loc)

        # h_benetos.set_visible(False)
        # h_duan.set_visible(False)
        # h_cnn.set_visible(False)
        h_piero.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(path, "all_melody_scores.pdf"), format='pdf', bbox_inches='tight')

# global path
# path = "/data/Experiments/SOFTMAX"
# plot(path)
