import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


balance_data = pd.read_csv('heart(1).csv')
balance_data.head()


X = balance_data.drop(columns=['target'])
Y = balance_data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0) 

clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 0)
clf_gini.fit(X_train, y_train) 

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
clf_entropy.fit(X_train, y_train) 

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred 

def cal_accuracy(y_test, y_pred):
    print("Accuracy : ", accuracy_score(y_test,y_pred)*100)

print("Results Using Gini:")

y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)

print("Results Using Entropy:")

y_pred_entropy = prediction(X_test, clf_entropy)
cal_accuracy(y_test, y_pred_entropy)

def printTree(classifier):
    feature_names = ['Chest Pain', 'Blood Circulation', 
                         'Blocked Arteries']
    target_names = ['HD-Yes', 'HD-No']
    

    dot_data = tree.export_graphviz(classifier,                                      
                         out_file=None,feature_names=feature_names,
                         class_names=target_names, filled = True)
    

    tr = graphviz.Source(dot_data, format ="png")
    return tr

plt.figure(figsize=(20,20))
features = balance_data.columns
classes = ['Not heart disease','heart disease']
tree.plot_tree(clf_gini,feature_names=features,class_names=classes,filled=True)
plt.show()

plt.figure(figsize=(20,20))
features = balance_data.columns
classes = ['Not heart disease','heart disease']
tree.plot_tree(clf_entropy,feature_names=features,class_names=classes,filled=True)
plt.show()