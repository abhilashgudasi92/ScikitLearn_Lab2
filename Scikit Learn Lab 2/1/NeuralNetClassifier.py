
# coding: utf-8


from sklearn.datasets import load_digits
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
digits = load_digits()
x = digits.data
y = digits.target
y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y.shape[1]
for i in range(0,5):
    print()
    print("round:",i)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
    classifier = MLPClassifier()
    from sklearn.grid_search import GridSearchCV
    parameters={
    'learning_rate':["constant", "invscaling", "adaptive"],
    'learning_rate_init':[0.01,0.1,0.5],
    'hidden_layer_sizes':[(10,8,6), (10,10,5), (8,3)],
    'alpha': [0.0001, 0.001,0.01,0.1],#10.0 ** -np.arange(1, 7)],
    'activation':["logistic", "relu", "tanh"]
    }
    MlpClf = GridSearchCV(classifier,param_grid=parameters)
    y_score = MlpClf.fit(X_train, y_train).predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc[1]

    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.metrics import classification_report
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print(classification_report(y_test, y_score))
    print()

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test.argmax(axis=1), y_score.argmax(axis=1))
    print(matrix)

    score = MlpClf.score(X_test, y_test)
    print(score)


    MlpClf.best_params_

    MlpClf.best_estimator_

    MlpClf.best_score_
