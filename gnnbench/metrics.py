from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def accuracy(ground_truth, predictions):
    return accuracy_score(ground_truth, predictions)


def f1(ground_truth, predictions):
    return f1_score(ground_truth, predictions, average='macro')


def precision(ground_truth, predictions):
    return precision_score(ground_truth, predictions, average='macro')


def recall(ground_truth, predictions):
    return recall_score(ground_truth, predictions, average='macro')


