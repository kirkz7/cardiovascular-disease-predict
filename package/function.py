import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
def eval_confusion(name,score,ypr,ytest):
    print("The score is %f for %s" % (score,name))
    cm = confusion_matrix(ytest, ypr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
def eval_auc(name,ytest,probTest):
    fpr, tpr, thresholds = roc_curve(ytest,probTest)
    auc = np.round(roc_auc_score(y_true=ytest,y_score=probTest),decimals=3)
    plt.plot(fpr, tpr, label="AUC - %s = " % name + str(auc))
    plt.legend(loc=4)
    plt.show()