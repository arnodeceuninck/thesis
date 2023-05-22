import matplotlib.pyplot as plt
import seaborn as sns

import textwrap


def wrap_labels(ax, width):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=True))
    ax.set_xticklabels(labels, rotation=0)


def plot_scores(scores, plot_title, baseline_name="Random Forest (NaNs dropped in train)", hue=None, output_file=None,
                reverse_axes=False, wrap=False, x="model", xlabel=None, ylabel=None,
                plot=None):
    if plot is None:
        should_plot = output_file is None
    else:
        should_plot = plot

    x_label_rotation = 90 if not wrap else 0

    x_label = x

    # plot starting at 0
    x = x_label
    y = "auc"
    x, y = (y, x) if reverse_axes else (x, y)
    plot = sns.barplot(x=x, y=y, data=scores, ci="sd", hue=hue)
    plot.set(title=plot_title)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=x_label_rotation)

    if wrap:
        wrap_labels(plot, 10)

    if xlabel:
        plot.set_xlabel(xlabel)
    if ylabel:
        plot.set_ylabel(ylabel)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')

    if should_plot:
        plt.show()

    plot.figure.clf()

    if baseline_name:
        # plot showing improvement from baseline
        baseline = scores[(scores['model'] == baseline_name) & (scores['chain'] == 'both')]['auc'].mean()
        scores["improvement"] = scores["auc"] - baseline

        x = x_label
        y = "improvement"
        x, y = (y, x) if reverse_axes else (x, y)
        plot = sns.barplot(x=x, y=y, data=scores, ci="sd", hue=hue)
        plot.set(title=f"{plot_title} (improvement)")
        plot.set_xticklabels(plot.get_xticklabels(), rotation=x_label_rotation)

        if wrap:
            wrap_labels(plot, 10)

        if xlabel:
            plot.set_xlabel(xlabel)
        if ylabel:
            plot.set_ylabel(ylabel)

        plot.axhline(0, color="k", clip_on=False) if not reverse_axes else plot.axvline(0, color="k", clip_on=False)
        y_ticks = plot.get_yticks()
        plot.set_yticklabels([f"{baseline + y:.2f}" for y in y_ticks])

        if output_file is not None:
            plt.savefig(f"{output_file}-improvement.png", bbox_inches='tight')

        if should_plot:
            plt.show()

        plot.figure.clf()
