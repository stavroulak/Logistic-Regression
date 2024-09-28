# If you need:
# !pip install scikit-learn==0.23.1

# Train a logistic regression model

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix

# Input as a csv file (all the classified data to train your model):
input = 'morphology.csv'
# The names of your columns which will be used to classify tha data
columns_x = ['log(W1)', 'log(NUV)', 'log(250)', 'n'] # , 'WISE_3.4' , 'SPIRE_250', 'GALEX_NUV', 'q', 'n'
# The name of your column with the classification (the categories to me numerical integers: e.g. 0 and 1)
column_y = 'type'


# Main

df = pd.read_csv(input)
df['log(W1)'] = np.log(df['WISE_3.4'].values)
df['log(NUV)'] = np.log(df['GALEX_NUV'].values)
df['log(250)'] = np.log(df['SPIRE_250'].values)
df = df.dropna(axis=0)

X = df[columns_x].values  #.astype(float)
y = df[column_y].values

# Normalize the data:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split your sample
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4) # test_size=0.2 means that 20% of your sample wil be the test sample

# Train the model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train) # lower C values: higher level of regulation
                                                                         # other solver options:  ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
# Predict
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# Evaluation metrics
print("Jaccard score:")
print(jaccard_score(y_test, yhat, pos_label=0))

print("Classification report:")
print(classification_report(y_test, yhat))

print("Train set Accuracy: ", metrics.accuracy_score(y_train, LR.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

df2= pd.DataFrame(y_test)
df2["predicted"] = yhat
df2["probability"] = y_test
df2.to_csv("accuracy.csv")


# Confusion matrix: (optional)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print(confusion_matrix(y_test, yhat, labels=[1,0]))

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.savefig("confusion_matrix.png")
