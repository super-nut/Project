import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        confusion_matrix = [[0, 0], [0, 0]]
        for t, p in zip(pred.view(-1), target.view(-1)):
            confusion_matrix[t.long()][p.long()] += 1
        assert pred.shape[0] == len(target)

    return confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1] + 1e-6)


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        confusion_matrix = [[0, 0], [0, 0]]
        for t, p in zip(pred.view(-1), target.view(-1)):
            confusion_matrix[t.long()][p.long()] += 1
        assert pred.shape[0] == len(target)

    return confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1] + 1e-6)


def f1(output, target):
    p = precision(output, target)
    r = recall(output, target)
    return 2 * p * r / (p + r + 1e-6)
