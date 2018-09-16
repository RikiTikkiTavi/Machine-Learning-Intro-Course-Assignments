import pandas as pd
import sklearn.metrics as metrics

scores_data = pd.read_csv('classification.csv')
y = scores_data[['true']].as_matrix()
predictions = scores_data[['pred']].as_matrix()


def write_values(values, name):
    f = open(name, 'w')
    delimiter = ""
    for val in values:
        val = str(val)
        f.write(delimiter + val)
        delimiter = " "


def calculate_mistake_matrix(predictions, true_classes):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    p_length = len(predictions)
    for i in range(p_length):
        p = predictions[i]
        cl = true_classes[i]
        if p == 1:
            if cl == 1:
                TP += 1
            else:
                FP += 1
        else:
            if cl == 0:
                TN += 1
            else:
                FN += 1
    return TP, FP, FN, TN


TP, FP, FN, TN = calculate_mistake_matrix(predictions, y)

print("TP", TP)
print("FP", FP)
print("FN", FN)
print("TN", TN)

write_values([TP, FP, FN, TN], '1.txt')

accuracy = metrics.accuracy_score(y, predictions)
precision = metrics.precision_score(y, predictions)
recall = metrics.recall_score(y, predictions)
f1_score = metrics.f1_score(y, predictions)

write_values([accuracy, precision, recall, f1_score], "2.txt")


# SCORES

def calculate_max_score(score_logreg, score_svm, score_knn, score_tree):
    scores = [
        metrics.roc_auc_score(y, score_logreg),
        metrics.roc_auc_score(y, score_svm),
        metrics.roc_auc_score(y, score_knn),
        metrics.roc_auc_score(y, score_tree),
    ]
    max_score = max(scores)
    index_of_max = scores.index(max(scores))
    return max_score, index_of_max


scores_data = pd.read_csv('scores.csv')

score_logreg = scores_data[["score_logreg"]].as_matrix()
score_svm = scores_data[["score_svm"]].as_matrix()
score_knn = scores_data[["score_knn"]].as_matrix()
score_tree = scores_data[["score_tree"]].as_matrix()
y = scores_data[['true']].as_matrix().ravel()

max_score, index_of_max = calculate_max_score(score_logreg, score_svm, score_knn, score_tree)
print(max_score)
write_values([list(scores_data.columns.values)[index_of_max + 1]], "3.txt")

# Max precision by >70% recall

pre_rec_curve = [
    metrics.precision_recall_curve(y, score_logreg),
    metrics.precision_recall_curve(y, score_svm),
    metrics.precision_recall_curve(y, score_knn),
    metrics.precision_recall_curve(y, score_tree),
]

max_precisions = {}

for i in range(len(pre_rec_curve)):
    metric = pre_rec_curve[i]
    for k in range(len(metric[1])):
        rec = metric[1][k]
        if rec > 0.7:
            max_precisions[i] = metric[0][k]

index = max(max_precisions, key=max_precisions.get)

write_values(([list(scores_data.columns.values)[index + 1]]), "4.txt")