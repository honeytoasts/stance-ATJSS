# 3rd-party module
import pandas as pd
from sklearn import metrics

def stance_score(targets, label_y, pred_y):
    consider_labels = [0, 1]  # only consider "favor (0)" and "against (1)"
    target_name = ['atheism', 'climate change is a real concern',
                   'feminist movement', 'hillary clinton',
                   'legalization of abortion']

    label_series, pred_series = pd.Series(label_y), pd.Series(pred_y)
    target_f1 = []

    # get f1-score for each target
    for target in target_name:
        labels = label_series[targets == target].tolist()
        preds = pred_series[targets == target].tolist()
        f1 = metrics.f1_score(labels, preds, average='macro', labels=consider_labels)

        target_f1.append(f1)

    # get macro-f1 and micro-f1
    macro_f1 = sum(target_f1) / len(target_f1)
    micro_f1 = metrics.f1_score(label_y,
                                pred_y,
                                average='macro',
                                labels=consider_labels,
                                zero_division=0)

    return target_f1, macro_f1, micro_f1

def sentiment_score(label_y, pred_y):
    labels = [0, 1, 2]

    f1 = metrics.f1_score(label_y,
                          pred_y,
                          average='macro',
                          labels=labels,
                          zero_division=0)

    return f1