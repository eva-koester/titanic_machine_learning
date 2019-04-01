import log_regession
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def cm_interpreter():
    """find out which numbers of confusion matrix are true/false positive and true/false negative"""
    y_test, y_pred = log_regession.logistic_regression()
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))

    # interpret confusion matrix
    # how many survived and dead
    survived = np.count_nonzero(y_pred)
    l = list(y_pred)
    dead = l.count(0)
    assert len(y_pred) == (survived + dead)

    # examine which values are true neg, true pos etc, assuming that true pos > false neg and true neg > false pos
    cm_list = list(cm)
    flat_list = []
    for sublist in cm_list:
        for item in sublist:
            flat_list.append(item)

    for (x, y) in [(x,y) for x in flat_list for y in flat_list]:
        if (x+y) == dead:
            if x > y:
                true_pos = x
                false_neg = y
            elif y > x:
                true_pos = y
                false_neg = x
        if (x+y) == survived:
            if x > y:
                true_neg = x
                false_pos = y
            elif y > x:
                true_neg = y
                false_pos = x
            break
    print('true_pos:', true_pos)
    print('false_neg:', false_neg)
    print('true_neg:', true_neg)
    print('false_pos:', false_pos)
