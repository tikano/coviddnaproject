print("heyyyy")

from Bio import SeqIO

print("started")

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import numpy as np
import pandas as pd
import tqdm

import seaborn as sns
import graphviz
#import pydot

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn import model_selection, linear_model
from sklearn.tree import DecisionTreeClassifier, plot_tree
from io import StringIO

print("here")

import gdown

print(0)

data_path = 'https://drive.google.com/uc?id=1f1CtRwSohB7uaAypn8iA4oqdXlD_xXL1'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'
gdown.download(data_path, cov2_sequences, True)

print(1)

sequences = [r for r in SeqIO.parse(cov2_sequences, 'fasta')]
sequence_num =  1
print(sequences[sequence_num])


print(2)


mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])


print(3)
 
for location in tqdm.tqdm(range(n_bases_in_seq)): # tqdm is a nice library that prints our progress.
  bases_at_location = np.array([s[location] for s in sequences])
  if len(set(bases_at_location))==1: continue 
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)
    
    feature_values[bases_at_location==['N']] = np.nan
    
    feature_values  = feature_values*1
    
    column_name = str(location) + '_' + base
    mutation_df[column_name] = feature_values

n_rows = np.shape(mutation_df)[0]
n_columns = np.shape(mutation_df)[1]
print("Size of matrix: %i rows x %i columns" %(n_rows, n_columns))

mutation_df.tail()



countries = [(s.description).split('|')[-1] for s in sequences]

	  
countries_to_regions_dict = {
         'Australia': 'Oceania',
         'China': 'Asia',
         'Hong Kong': 'Asia',
         'India': 'Asia',
         'Nepal': 'Asia',
         'South Korea': 'Asia',
         'Sri Lanka': 'Asia',
         'Taiwan': 'Asia',
         'Thailand': 'Asia',
         'USA': 'North America',
         'Viet Nam': 'Asia'
}

regions = [countries_to_regions_dict[c] if c in 
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions

balanced_df = mutation_df.copy()
balanced_df['label'] = regions
balanced_df = balanced_df[balanced_df.label!='NA']
balanced_df = balanced_df.drop_duplicates()
samples_north_america = balanced_df[balanced_df.label== ####### FILL IN ####
                                    'North America']
samples_oceania = balanced_df[balanced_df.label== ##### FILL IN #########
                              'Oceania']
samples_asia = balanced_df[balanced_df.label== ##### FILL IN #######
                           'Asia']

print(4)

# Number of samples we will use from each region.
n = min(len(samples_north_america),
        len(samples_oceania),
        len(samples_asia))

balanced_df = pd.concat([samples_north_america[:n],
                    samples_asia[:n],
                    samples_oceania[:n]])
print("Number of samples in each region: ", Counter(balanced_df['label']))

X = balanced_df.drop('label', 1)
Y = balanced_df.label
data = "X (features)" #@param ['X (features)', 'Y (label)']
start = 1 #@param {type:'integer'}
stop =  10#@param {type:'integer'}

print(5)

if start>=stop:print("Start must be < stop!")
else:
  if data=='X (features)':
    print(X.iloc[start:stop])
  if data=='Y (label)':
    print(Y[start:stop])

lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Split into training/testing set.
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, train_size=.8)

print(6)

# Train/fit model.
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

# Compute accuracy.
accuracy = 100*np.mean(Y_pred==Y_test)
print("Accuracy: %", accuracy)

# Compute confusion matrix.
confusion_mat = pd.DataFrame(confusion_matrix(Y_test, Y_pred))
confusion_mat.columns = [c + ' predicted' for c in lm.classes_]
confusion_mat.index = [c + ' true' for c in lm.classes_]

print(confusion_mat)

lam = 4
l1m = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, C=1/lam,
    penalty='l1', solver='saga', random_state=42)
l1m.fit(X_train, Y_train)

print("Training accuracy:", np.mean(Y_train==l1m.predict(X_train)))
print("Testing accuracy:", np.mean(Y_test==l1m.predict(X_test)))
print("Number of non-zero coefficients in lasso model:", sum(l1m.coef_[0]!=0))

clf = DecisionTreeClassifier(random_state=0, max_depth=9)
clf.fit(X_train, Y_train)

plt.figure()
plot_tree(clf, filled=True)
plt.savefig("DecisionTree.png",bbox_inches='tight',dpi=300)
plt.show()

