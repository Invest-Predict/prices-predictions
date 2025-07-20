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

def f1_metric_long(y_pred, close, commission):
    pr = precision_long(y_pred, close, commission)
    rec = recall_long(y_pred, close, commission)
    return 2 * pr * rec / (pr + rec) if (pr + rec) > 0 else 0
