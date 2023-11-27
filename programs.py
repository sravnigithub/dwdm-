import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Load ARFF data into a Pandas DataFrame
data, meta = arff.loadarff('/home/sravani/Documents/student.arff')
df = pd.DataFrame(data)
categorical_columns = ['age', 'income', 'student',  'buyspc']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

x = df.drop('buyspc', axis=1, errors='ignore')
y = df['buyspc']

print(x)
print(y)
print(label_encoders)





import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Load ARFF data into a Pandas DataFrame
data, meta = arff.loadarff('/home/sravani/Downloads/labor.arff')
df = pd.DataFrame(data)

# Print column names to check for typos or discrepancies
print(df.columns)

categorical_columns = ['cost_of_living', 'adjustment', 'pension', 'education_allowance', 'vacation', 'class', 'longterm_disability']
label_encoders = {}

for column in categorical_columns:
    # Check if the column is present in the DataFrame
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    else:
        print(f"Column '{column}' not found in the DataFrame.")

# Check if 'class' column is present before accessing it
if 'class' in df.columns:
    x = df.drop('class', axis=1, errors='ignore')
    y = df['class']
    print(x)
    print(y)
else:
    print("Column 'class' not found in the DataFrame.")







from scipy.io import arff

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

 

# Load the ARFF file

data = arff.loadarff('/home/sravani/Downloads/contact-lens.arff')

df = pd.DataFrame(data[0])

 

# Convert the nominal attributes to strings

for col in df.columns:

    if pd.api.types.is_categorical_dtype(df[col]):

        df[col] = df[col].str.decode('utf-8')

 

# Convert the dataset into a one-hot encoded format

oht = pd.get_dummies(df.iloc[:, :-1], columns=df.columns[:-1], prefix='', prefix_sep='')

 

# Find frequent itemsets using the Apriori algorithm

min_support = 0.2  # Minimum support threshold (adjust as needed)

frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)

 

# Display frequent itemsets

print("Frequent Itemsets:")

print(frequent_itemsets)

 

# Find association rules

min_confidence = 0.7  # Minimum confidence threshold (adjust as needed)

association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)

 

# Display association rules

print("\nAssociation Rules:")

print(association_rules_df)







import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
data = pd.read_csv("/home/sravani/Downloads/super market.csv")

# Convert 'y' and 'n' to boolean values (True and False)
data = data.applymap(lambda x: True if x == 'y' else False)

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(data)

# Find frequent item sets with minimum support
frequent_itemsets = apriori(one_hot_encoded, min_support=0.2, use_colnames=True)

# Generate association rules with minimum confidence and compute lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display the association rules
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Save the rules to a CSV file if needed
# rules.to_csv("association_rules.csv", index=False)





from scipy.io import arff
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

data, meta = arff.loadarff("/home/sravani/Documents/student.arff")

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                       random_state=1)

kf = KFold(n_splits=10, random_state=1, shuffle=True)
cross_val_results = cross_val_score(dt_classifier, X, y, cv=kf)

print("Cross-validation scores:", cross_val_results)
print("Mean accuracy:", cross_val_results.mean())
dt_classifier.fit(X, y)
y_pred = dt_classifier.predict(X)
print("\nClass-wise statistics:")
print(classification_report(y, y_pred))






import csv
import re
from scipy.io import arff
import pandas as pd

# Load ARFF data into a Pandas DataFrame
data, meta = arff.loadarff('/home/sravani/Downloads/employee.arff')
df = pd.DataFrame(data)

# Function to clean and convert columns to appropriate data types
def clean_and_convert(df, attribute):
    if attribute in df.columns:
        if df[attribute].dtype == 'object':
            df[attribute] = df[attribute].str.decode('utf-8')  # Convert bytes to string
            df[attribute] = df[attribute].apply(lambda x: re.sub(r"[^a-zA-Z0-9_]+", "", x))  # Remove special characters
        df[attribute] = df[attribute].astype(str)

# Clean and convert each attribute
for attribute in df.columns:
    clean_and_convert(df, attribute)

# List of attributes (excluding the target attribute)
attributes = ['age', 'salary']

# Target attribute
target_attribute = 'performance'

# Convert 'age' and 'salary' columns to integers
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].apply(lambda x: re.sub(r'\D', '', x)).astype(int)

# Build the decision tree
decision_tree = id3(df.to_dict(orient='records'), attributes, target_attribute)

# Print the resulting decision tree
import pprint
pprint.pprint(decision_tree)






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset into a Pandas DataFrame
data = pd.read_csv('/home/sravani/Downloads/employee.csv')

# Encode the categorical attributes (age, salary, performance) to numeric values
le = LabelEncoder()
data['age'] = le.fit_transform(data['age'])
data['salary'] = le.fit_transform(data['salary'])
data['performance'] = le.fit_transform(data['performance'])

# Split the dataset into features (X) and target (y)
X = data[['age', 'salary']]
y = data['performance']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)






import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
data = pd.read_csv("/home/sravani/Downloads/Dataset/iris.csv")  # Replace "your_dataset.csv" with the actual file path

# Select the features for clustering
X = data.iloc[:, :-1].values

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph to find the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster-Sum-of-Squares
plt.show()

# Based on the Elbow method, choose the optimal number of clusters (e.g., 3)
optimal_num_clusters = 3

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_clusters = kmeans.fit_predict(X)

# Add the cluster labels to the DataFrame
data['Cluster'] = pred_clusters

# Print the results
print(data)

# Visualize the clusters (for 2D data)
plt.scatter(X[pred_clusters == 0, 0], X[pred_clusters == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[pred_clusters == 1, 0], X[pred_clusters == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[pred_clusters == 2, 0], X[pred_clusters == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()





import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the ARFF data
data = arff.loadarff("/home/sravani/Documents/student.arff")  # Replace "student.arff" with the actual file path
df = pd.DataFrame(data[0])

# Convert categorical attributes to numerical values
df['age'] = df['age'].map({b'<30': 1, b'30-40': 2, b'>40': 3})
df['income'] = df['income'].map({b'low': 1, b'medium': 2, b'high': 3})
df['student'] = df['student'].map({b'no': 0, b'yes': 1})
df['credit-rating'] = df['credit-rating'].map({b'fair': 1, b'excellent': 2})
df['buyspc'] = df['buyspc'].map({b'no': 0, b'yes': 1})

# Select the features for clustering
X = df.iloc[:, :-1].values

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph to find the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster-Sum-of-Squares
plt.show()

# Based on the Elbow method, choose the optimal number of clusters (e.g., 3)
optimal_num_clusters = 3

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_clusters = kmeans.fit_predict(X)

# Add the cluster labels to the DataFrame
df['Cluster'] = pred_clusters

# Visualize the clusters (for 2D data)
feature1_index = 0
feature2_index = 1

for cluster_num in range(optimal_num_clusters):
    plt.scatter(X[pred_clusters == cluster_num, feature1_index], X[pred_clusters == cluster_num, feature2_index], label=f'Cluster {cluster_num + 1}')

plt.scatter(kmeans.cluster_centers_[:, feature1_index], kmeans.cluster_centers_[:, feature2_index], s=100, c='black', label='Centroids')
plt.title('Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Print the results
print(df)
