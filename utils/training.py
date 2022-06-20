import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, pos_label='positive', average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, pos_label='positive', average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, pos_label='positive', average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
