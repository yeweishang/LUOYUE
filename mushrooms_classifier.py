import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.tree.export import export_text
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree.export import export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, classification_report

def print_metrics(labels_test, predictions, title):
    
    print(f'Metrics for {title}')
    print('Precision: ', precision_score(labels_test, predictions))
    print('Recall: ', recall_score(labels_test, predictions))
    print('Accuracy:', accuracy_score(labels_test, predictions))
    print('Average Precision: ', average_precision_score(labels_test, predictions))
    # create_confusion_matrix(labels_test, predictions, title)
    print('')

def create_confusion_matrix(labels_test, predictions, title):
    classes = ['Edible','Poisonus']
    cnf_matrix =  confusion_matrix(labels_test, predictions)
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix for {title}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
        
    #cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    title_striped = title.strip()
    plt.savefig(f'images/cm_{title_striped}.png')




def classify_by_decision_tree(Features_train, labels_train, Features_test, labels_test):
    # #Decision Tree
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(Features_train, labels_train)
    predictions_dt = dt.predict(Features_test)
    predictions_dt = predictions_dt.reshape((-1, 1))
    
    print_metrics(labels_test, predictions_dt, f'Decision Tree')
    create_confusion_matrix(labels_test, predictions_dt, f'Decision Tree')

'''def classify_by_multi_layer_perceptron(Features_train, labels_train, Features_test, labels_test):
    #Multi-Layer Perceptron
    mlp = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 5), max_iter=1000)
    mlp.fit(Features_train, labels_train.values.ravel())
    predictions_mlp = mlp.predict(Features_test)

    print_metrics(labels_test, predictions_mlp, 'Multi-Layer Perceptron')
    create_confusion_matrix(labels_test, predictions_mlp, 'Multi-Layer Perceptron')'''
    

def classify_by_bayesian(Features_train, labels_train, Features_test, labels_test):
    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(Features_train, labels_train.values.ravel())
    predictions_gnb = gnb.predict(Features_test)

    print_metrics(labels_test, predictions_gnb,f'Gaussian Naive Bayes')
    create_confusion_matrix(labels_test, predictions_gnb, f'Gaussian Naive Bayes')

def classify_by_svm(Features_train, labels_train, Features_test, labels_test):
    #Support Vector Machines
    svc = svm.SVC(gamma='scale')
    svc.fit(Features_train, labels_train.values.ravel())
    predictions_svc = svc.predict(Features_test)

    print_metrics(labels_test, predictions_svc, f'Support Vector Machines')
    create_confusion_matrix(labels_test, predictions_svc, f'Support Vector Machines')

def classify_by_knn(Features_train, labels_train, Features_test, labels_test):
    #K Nearest Neighbours
    for i in range(5,21):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(Features_train, labels_train.values.ravel())
        predictions_knn = knn.predict(Features_test)
        print_metrics(labels_test, predictions_knn,f'K Nearest Neighbours - k = {i}')
        create_confusion_matrix(labels_test, predictions_knn, f'K Nearest Neighbours - k = {i}')


if __name__ == "__main__":
    Features_train = pd.read_csv("data/Features_train.csv")
    Features_test = pd.read_csv("data/Features_test.csv")
    labels_train = pd.read_csv("data/labels_train.csv")
    labels_test = pd.read_csv("data/labels_test.csv")
    # classify_by_decision_tree(Features_train, labels_train, Features_test, labels_test)
    # classify_by_multi_layer_perceptron(Features_train, labels_train, Features_test, labels_test)
    #classify_by_bayesian(Features_train, labels_train, Features_test, labels_test)
    classify_by_svm(Features_train, labels_train, Features_test, labels_test)
    # classify_by_knn(Features_train, labels_train, Features_test, labels_test)

