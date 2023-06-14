import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd

# warnings
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings("ignore",
                        category=UserWarning)  # hide: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

import textwrap

def sort_df(df, attributes_to_sort_by):
    """
    WARNING: A dict has no order, so only one attribute to sort by is recommended

    Sort the dataframe based on the given attributes and the order in their list
    :param df: Dataframe to sort
    :param attributes_to_sort_by: List of attributes to sort by (dict, keys are the attributes, values are the order)
    :return: Sorted dataframe
    """
    df_len = len(df)

    df = df.copy()

    # sort them in reverse order, so they will finally be sorted on the first attribute
    for key, value in reversed(attributes_to_sort_by.items()):

        new_df = pd.DataFrame()
        for v in value:
            new_df = pd.concat([new_df, df[df[key] == v]])

        # add the rest of the dataframe
        new_df = pd.concat([new_df, df[~df[key].isin(value)]])

        df = new_df

    assert len(df) == df_len, "Sorting went wrong, length of dataframe changed"
    return df


def compose_all(inputs, output=None, models_to_keep=None, attributes_to_keep=None):
    """
    Compose all the data from the different models into one file
    :param inputs: Array of score csv files to combine
    :param output: Output file
    :param models_to_keep: List of strings containing the names of the models to keep (of None, keep all)
    :param attributes_to_keep: Dict of list where the key is a column name and the value is a list of values to keep
    :return:
    """
    attributes_to_keep = {} if attributes_to_keep is None else attributes_to_keep

    if models_to_keep is not None:
        attributes_to_keep['model'] = models_to_keep

    df = pd.DataFrame()

    for i in inputs:
        df = pd.concat([df, pd.read_csv(i)])

    if attributes_to_keep is not None:
        for key, value in attributes_to_keep.items():
            df = df[df[key].isin(value)]

        # sort in the order of the (categorical) lists in attributes_to_keep
        df = df.sort_values(by=list(attributes_to_keep.keys()))

    if 'model' in attributes_to_keep:
        df = sort_df(df, {'model': attributes_to_keep['model']})

    df.to_csv(output, index=False) if output is not None else None

    return df


def wrap_labels(ax, width):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=True))
    ax.set_xticklabels(labels, rotation=0)


def plot_scores(scores, plot_title, baseline_name="Random Forest (NaNs dropped in train)", hue=None, output_file=None,
                reverse_axes=False, wrap=False, x="model", xlabel=None, ylabel=None,
                plot=None, hue_group_diff=True, improvement_config=None):
    """

    :param scores:
    :param plot_title:
    :param baseline_name:
    :param hue:
    :param output_file:
    :param reverse_axes:
    :param wrap:
    :param x:
    :param xlabel:
    :param ylabel:
    :param plot:
    :param hue_group_diff:
    :param improvement_config: dict containing  'value' and optional 'legend outside'
    :return:
    """
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

    if improvement_config is not None:
        improvement_variable = improvement_config['value']
        df_true = scores[scores[improvement_variable] == True].copy()
        df_false = scores[scores[improvement_variable] == False].copy()

        plot = sns.barplot(x=x, y=y, data=df_true, ci="sd", hue=hue, alpha=0.4)
        plot = sns.barplot(x=x, y=y, data=df_false, ci="sd", hue=hue, alpha=0.7)

        df_true_models = df_true['model'].unique()
        df_false_models = df_false['model'].unique()

        # plot the hidden bars again
        # replot = False
        # for i in range(len(df_true_models)):
        #     true_mean = df_true[df_true['model'] == df_true_models[i]]['auc'].mean()
        #     false_mean = df_false[df_false['model'] == df_false_models[i]]['auc'].mean()
        #
        #     if true_mean > false_mean:
        #         # don't need to replot, so set score to 0
        #         df_true.loc[df_true['model'] == df_true_models[i], 'auc'] = 0
        #     else:
        #         # need to replot
        #         replot = True
        #
        # if replot:
        #     # plot on top of previous plot (so it becomes visible)
        #     plot = sns.barplot(x=x, y=y, data=df_true, ci="sd", hue=hue, alpha=0.4)


        legend_outside = improvement_config.get('legend outside', False)

        # place legend at left top outside of plot
        plot.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

        # if legend_outside:
        #     # Does not work for some reason
        #     sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))




    else:

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
        # check if baseline is in scores, else take first name
        if baseline_name not in scores['model'].unique():
            new_baseline_name = scores['model'].unique()[0]
            print(f"Baseline {baseline_name} not found in scores, using {new_baseline_name} instead")
            baseline_name = new_baseline_name
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
