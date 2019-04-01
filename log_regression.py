import pandas as pd
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# preprocessing steps
df = preprocessing.preprocess_data()
y = df['survived'].values
X = df.drop('survived', axis=1).values
assert len(X) == len(y)

# use log regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

# interpret confusion matrix
# how many diabetes and no diabetes
# diabetes = np.count_nonzero(y_pred)
# print(diabetes)
#
# l = list(y_pred)
# no_diabetes = l.count(0)
# print(no_diabetes)
#
# # assert that array only consists of 0 and 1:
# assert len(y_pred) == (diabetes + no_diabetes)
#
# print(cm[0,0])
# print(cm[1,0])
# # print(cm[0]).sum
# print(np.add(cm[0,0], cm[0,1]))
# print(no_diabetes)
#
# # examine which values are true neg, true pos etc, assuming that true pos > false neg and true neg > false pos
# cm_list = list(cm)
# flat_list = []
# for sublist in cm_list:
#     for item in sublist:
#         flat_list.append(item)
#
# for (x, y) in [(x,y) for x in flat_list for y in flat_list]:
#     if (x+y) == no_diabetes:
#         if x > y:
#             true_pos = x
#             false_neg = y
#         elif y > x:
#             true_pos = y
#             false_neg = x
#     if (x+y) == diabetes:
#         if x > y:
#             true_neg = x
#             false_pos = y
#         elif y > x:
#             true_neg = y
#             false_pos = x
#         break
# print('true_pos:', true_pos)
# print('false_neg:', false_neg)
# print('true_neg:', true_neg)
# print('false_pos:', false_pos)