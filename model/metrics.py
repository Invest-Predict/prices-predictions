def true_positive(y_pred, close, commission):
    return ((y_pred == 1) & (close.shift(-1) / close > 1 + commission)).sum()

def false_positive(y_pred, close, commission):
    return ((y_pred == 1) & (close.shift(-1) / close <= 1 + commission)).sum()

def false_negative(y_pred, close, commission):
    return ((y_pred == 0) & (close.shift(-1) / close > 1 + commission)).sum()

def precision_long(y_pred, close, commission):
    tp = true_positive(y_pred, close, commission)
    fp = false_positive(y_pred, close, commission)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_long(y_pred, close, commission):
    tp = true_positive(y_pred, close, commission)
    fn = false_negative(y_pred, close, commission)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_metric_long(y_pred, close, commission, beta=0.005):
    pr = precision_long(y_pred, close, commission)
    rec = recall_long(y_pred, close, commission)
    return (1 + beta * beta) * pr * rec / (beta * beta * pr + rec) if (beta * beta * pr + rec) > 0 else 0


def true_positive_short(y_pred, close, commission):
    return ((y_pred == 1) & (close / close.shift(-1) < 1 + commission)).sum() # P = 1: close / close.shift(-1) < 1 + commission

def true_negative_short(y_pred, close, commission):
    return ((y_pred == 0) & (close / close.shift(-1) >= 1 + commission)).sum() # N = 0 : close / close.shift(-1) >= 1 + commission 

def false_positive_short(y_pred, close, commission):
    return ((y_pred == 1) & (close / close.shift(-1) >= 1 + commission)).sum()

def false_negative_short(y_pred, close, commission):
    return ((y_pred == 0) & (close / close.shift(-1) < 1 + commission)).sum()


def precision_short(y_pred, close, commission):
    tn = true_negative_short(y_pred, close, commission)
    fn = false_negative_short(y_pred, close, commission)
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def recall_short(y_pred, close, commission):
    tn = true_negative_short(y_pred, close, commission)
    fp = false_positive_short(y_pred, close, commission)
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def f1_metric_short(y_pred, close, commission, beta=0.005):
    pr = precision_short(y_pred, close, commission)
    rec = recall_short(y_pred, close, commission)
    return (1 + beta * beta) * pr * rec / (beta * beta * pr + rec) if (beta * beta * pr + rec) > 0 else 0
