"""Score any set of "wait" / "book now" predictions the same way."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def score(name, y_true, y_pred):
    """Compare predictions (1 = wait, 0 = book now) with what really happened.

    precision: when it says "wait", how often the price really dropped
    recall:    of all real drops, how many it caught
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rule": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "says_wait": y_pred.mean(),     # share of snapshots where it says "wait"
        "right_waits": tp,              # said wait, price dropped
        "wasted_waits": fp,             # said wait, price did NOT drop
        "missed_drops": fn,             # said book now, but price dropped
    }