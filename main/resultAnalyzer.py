import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from util import readFromCsv, evaluateAdaptations


font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False, legendInside=False):
    columns = data.columns
    nColumns = len(columns)
    print("Columns:", columns)
    print("Data shape:", data.shape)
    fig = plt.figure()  # plt.figure(figsize=(10, 10 * nColumns/2))
    ax1 = fig.add_subplot(111)  # (nColumns, 1, 1)

    # Creating axes instance
    bp = ax1.boxplot(data, patch_artist=True,
                     notch='True', vert=True)

    colors = plt.cm.Spectral(np.linspace(.1, .9, 3))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(nColumns//3):
        c = np.append(c, colors, axis=0)

    colors = c

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    if columnNames is not None and len(columnNames) > 1:
        ax1.xaxis.set_ticks(np.arange(1.5, len(columnNames) * 2, step=2), columnNames)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # y-axis
    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min())/8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s'
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    # legend
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]], ["NSGA-III", "XDA", "Anchors"],)
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]], ["NSGA-III", "XDA", "Anchors"],
                   ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    # Adding title
    plt.title(name)

    # Removing top axes and right axes
    # ticks
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    """
    for i in range(int(nColumns/2)):
        i2 = i + int(nColumns/2)
        axn = fig.add_subplot(nColumns, 1, i + 2)
        subset = data[[columns[i], columns[i2]]]
        subset = subset.sort_values(columns[i2])
        subset = subset.reset_index(drop=True)
        # axn.title.set_text(columns[i] + ' | ' + columns[i + int(nColumns/2)])
        subset.plot(ax=axn, color=colors[[i, i2]])
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        fig.show()
    else:
        plt.clf()


def personalizedBarChart(data, name, path=None, show=False, percentage=False):
    colors = plt.cm.Spectral(np.linspace(.1, .9, 3))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(len(data.values) // 3):
        c = np.append(c, colors, axis=0)

    colors = c

    ax = data.plot.bar(title=name, color=colors)

    if len(data.index) > 1:
        plt.xticks(rotation=0)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.set_ylim(0, 1)
    if percentage:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for container in ax.containers:
        if percentage:
            values = ['{:.1%}'.format(v) for v in container.datavalues]
        else:
            values = ['{:.2}'.format(v) for v in container.datavalues]
        ax.bar_label(container, values, fontsize=10)

    """
    for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 5,
            f"{rect.get_height() * 100:.1f}%", fontsize=7,
            ha='center', va='bottom')
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()

os.chdir(sys.path[0])
evaluate = False

pathToResults = "../results/" #sys.argv[1]

featureNames = ["cruise speed",
                "image resolution",
                "illuminance",
                "controls responsiveness",
                "power",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]

reqs = ["req_0", "req_1", "req_2", "req_3"]
reqsNamesInGraphs = ["R1", "R2", "R3", "R4"]

# read dataframe from csv
results = readFromCsv(pathToResults + 'results.csv')
nReqs = len(results["nsga3_confidence"][0])
reqs = reqs[:nReqs]
reqsNamesInGraphs = reqsNamesInGraphs[:nReqs]
targetConfidence = np.full((1, nReqs), 0.8)[0]

if evaluate:
    evaluateAdaptations(results, featureNames)

# read outcomes from csv
customOutcomes = pd.read_csv(pathToResults + 'customDataset.csv')
nsga3Outcomes = pd.read_csv(pathToResults + 'nsga3Dataset.csv')
anchorsOutcomes = pd.read_csv(pathToResults + 'anchorsDataset.csv')

# build indices arrays
nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
anchorsConfidenceNames = ['anchors_confidence_' + req for req in reqs]
anchorsOutcomeNames = ['anchors_outcome_' + req for req in reqs]

#outcomes dataframe
outcomes = pd.concat([nsga3Outcomes[reqs], customOutcomes[reqs], anchorsOutcomes[reqs]], axis=1)
#outcomes.columns = np.append(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames)
outcomes.columns = np.array(nsga3OutcomeNames + customOutcomeNames + anchorsOutcomeNames)

outcomes = outcomes[list(sum(zip(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames), ()))]

# decompose arrays columns into single values columns
nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
anchorsConfidences = pd.DataFrame(results['anchors_confidence'].to_list(),
                                  columns=anchorsConfidenceNames)

# select sub-dataframes to plot
confidences = pd.concat([nsga3Confidences, customConfidences, anchorsConfidences], axis=1)
confidences = confidences[list(sum(zip(nsga3Confidences.columns, customConfidences.columns, anchorsConfidences.columns), ()))]
scores = results[["nsga3_score", "custom_score", "anchors_score"]]
times = results[["nsga3_time", "custom_time", "anchors_time"]]

# plots
plotPath = pathToResults + 'plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlot(confidences, "Confidences comparison", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True)

# predicted successful adaptations
nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
anchorsPredictedSuccessful = (confidences[anchorsConfidenceNames] > targetConfidence).all(axis=1)

personalizedBoxPlot(confidences[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)

print("NSGA-III predicted success rate: " + "{:.2%}".format(nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("Anchors predicted success rate: " + "{:.2%}".format(anchorsPredictedSuccessful.sum() / anchorsPredictedSuccessful.shape[0]))
print(str(anchorsConfidences.mean()) + "\n")

print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("Anchors mean probas of predicted success: \n" + str(anchorsConfidences[anchorsPredictedSuccessful].mean()) + '\n')

# predicted successful adaptations
nsga3Successful = outcomes[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
anchorsSuccessful = outcomes[anchorsOutcomeNames].all(axis=1)

nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
anchorsSuccessRate = anchorsSuccessful.mean()

# outcomes analysis
print("NSGA-III success rate: " + "{:.2%}".format(nsga3SuccessRate))
print(str(outcomes[nsga3OutcomeNames].mean()) + "\n")
print("XDA success rate:  " + "{:.2%}".format(customSuccessRate))
print(str(outcomes[customOutcomeNames].mean()) + "\n")
print("Anchors success rate: " + "{:.2%}".format(anchorsSuccessRate))
print(str(outcomes[anchorsOutcomeNames].mean()) + "\n")

successRateIndividual = pd.concat([outcomes[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomes[anchorsOutcomeNames].rename(columns=dict(zip(anchorsOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividual.columns = ['NSGA-III', 'XDA', 'Anchors']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRate = pd.DataFrame([[nsga3SuccessRate, customSuccessRate, anchorsSuccessRate]], columns=["NSGA-III", "XDA", "Anchors"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomes[customOutcomeNames][customPredictedSuccessful].all(axis=1).mean(),
                                               outcomes[anchorsOutcomeNames][anchorsPredictedSuccessful].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA", "Anchors"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)
