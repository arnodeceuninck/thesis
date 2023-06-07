import matplotlib.pyplot as plt
import numpy as np
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
                plot=None, hue_group_diff=True):
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
        if not hue_group_diff:
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

        else:
            # Create a plot similar to the one above, but with the difference between different hue groups (horizontal next to each other)
            baseline_per_hue_group = scores[scores['model'] == baseline_name].groupby(x)['auc'].mean()
            scores["improvement_hue"] = scores.apply(lambda row: row["auc"] - baseline_per_hue_group[row[x]], axis=1)

            x = x_label
            y = "improvement_hue"
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

            # Change bar bottoms to the baseline of each hue group
            baseline_values = scores[scores['model'] == baseline_name].groupby(x)["auc"].mean()
            lowest_baseline = baseline_values.min()
            baseline_update = baseline_values - lowest_baseline

            # bar_containers = plot.containers
            # for container in bar_containers:
            #     bars = container.patches
            #     for i, bar, baseline_value in zip(range(len(bars)), bars, baseline_update):
            #         bar.set_y(baseline_value)

            new_lines = []
            bar_containers = plot.containers
            for container in bar_containers:
                bars = container.patches
                for i, bar, shift_value in zip(range(len(bars)), bars, baseline_update):
                    bar.set_y(shift_value)

                    # update all vertical lines in the range (the error bars)
                    # line because of a neighboring bar
                    eps = 0.0001
                    min_x = bar.get_x() - eps
                    max_x = bar.get_x() + bar.get_width() + eps

                    # # draw a small red, vertical line on bar.get_x
                    # plot.axvline(bar.get_x(), color="r", clip_on=False)

                    lines = plot.lines
                    # found = False
                    for line in lines:
                        if min_x <= line.get_xdata()[0] <= max_x:
                            # print(f"{min_x} <= {line.get_xdata()[0]} < {max_x}")
                            new_bottom = line.get_ydata()[0] + shift_value
                            new_top = line.get_ydata()[1] + shift_value
                            line.set_ydata([new_bottom, new_top])

                            # if found:
                            #     print(f"Found multiple lines for bar {i} ({min_x} <= x < {max_x}), current line: {line.get_xdata()[0]}")
                            # found = True
                            # break

                    # if not found:
                    #     print(f"Could not find line for bar {i} ({min_x} <= x < {max_x})")
                    #     closest_line = None
                    #     closest_distance = float("inf")
                    #     for line in lines:
                    #         distance = abs(line.get_xdata()[0] - min_x)
                    #         if distance < closest_distance:
                    #             closest_line = line
                    #             closest_distance = distance
                    #     print(f"Closest line is {closest_line.get_xdata()[0]}")

                    # draw a horizontal line below the bar (which is the length of the bar size), make it dotted if no bar
                    bar_height = bar.get_height()
                    if np.isnan(bar_height):
                        line = plt.Line2D((bar.get_x(), bar.get_x() + bar.get_width()), (shift_value, shift_value), lw=1.5,
                                         linestyle=(0, (1, 1)), color='k')
                    else:
                        line = plt.Line2D((bar.get_x(), bar.get_x() + bar.get_width()), (shift_value, shift_value), lw=1.5,
                                          color='k')
                    new_lines.append(line)

            for line in new_lines:
                plot.add_line(line)

            # update figure so everything is visible
            max_update = baseline_update.max()
            current_ylim = plot.get_ylim()
            plot.set_ylim(current_ylim[0], current_ylim[1] + max_update)

            y_ticks = plot.get_yticks()
            # plot.set_yticklabels([f"{lowest_baseline + y:.2f}" if 0 <= y + lowest_baseline <= 1 else "" for y in y_ticks])
            plot.set_yticklabels([f"{lowest_baseline + y:.2f}" for y in y_ticks])

            if output_file is not None:
                plt.savefig(f"{output_file}-improvement-per-group.png", bbox_inches='tight')

            if should_plot:
                plt.show()

            plot.figure.clf()
