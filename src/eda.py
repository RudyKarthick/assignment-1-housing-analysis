import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from random import sample


# Load cleaned data
df = pd.read_csv("housing_cleaned_final.csv")
print("Loaded cleaned dataset for EDA.")

print("Summary stats for features:")
print(df.describe())
df.describe().to_csv("eda_summary_stats.csv")


# -----Histogram-----
plt.hist(df['median_house_value'], bins=50, edgecolor='black')
plt.title("Distribution of Median House Value")
plt.ylabel("Number of Houses")
plt.xlabel("House Value ($)")

plt.grid(True)
plt.tight_layout()
plt.savefig("eda_median_house_value_hist.png")
print("Saved histogram to 'eda_median_house_value_hist.png'")
plt.show()




# -----Heatmap-----
# Source: https://commons.wikimedia.org/wiki/File:Map_of_California_outline.svg#filelinks
img = mpimg.imread("Map_of_California_outline.png")
print("Loaded California map.")

fig, ax = plt.subplots(figsize=(8,10))
# Lon & Lat bounds: https://en.wikipedia.org/wiki/California 
# Latitude	32°32′ N to 42° N = 32.533...° to 42°
# Longitude	114°8′ W to 124°26′ W = -114.133...° to -124.433...°
ax.imshow(img, extent=[-124.433, -114.133, 32.533, 42], aspect='auto')

sns.kdeplot(
    x=df['longitude'],
    y=df['latitude'],
    cmap='inferno',
    fill=True,
    alpha=0.5,
    ax=ax
)

plt.title("California Housing Data Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)
plt.tight_layout()
plt.savefig("eda_california_heatmap.png")
print("Saved geographic heatmap to 'eda_california_heatmap.png'")
plt.show()

# -----Scatterplot-----
sns.scatterplot(data=df, x='median_income', y='median_house_value')
plt.title("House Value and Median Income")
plt.xlabel("Income ($10k)")
plt.ylabel("House Value ($)")
plt.tight_layout()
plt.savefig("eda_house_value_and_median_income_scat.png")
plt.show()

# -----Bar Plot-----
mean = df.groupby('housing_median_age')['median_house_value'].mean()
plt.figure(figsize=(15,10))
mean.plot(kind='bar')
plt.title("Mean House Value & Housing Median Age")
plt.xlabel("Median House Age")
plt.ylabel("Median House Value ($)")
plt.tight_layout()
plt.savefig("eda_house_value_and_housing_age_bar.png")
plt.show()


# ---- Linear Regression from Scratch ----
print("Running Linear Regression: Predicting house value based on income...")

# Grab the two columns we're interested in: income and house value
X = df['median_income'].values
y = df['median_house_value'].values

# Add a column of 1s to X for the "intercept" term in y = mx + b
X_with_bias = np.c_[np.ones(len(X)), X]  # now each row is [1, income]

# Use the Normal Equation to calculate the best-fitting line:
# This is a formula to find the optimal slope and intercept
theta_best = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

intercept, slope = theta_best
print(f"Our model says: House Value = {intercept:.2f} + {slope:.2f} × Income")

# Let's visualize it!
plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.3, label="Actual Data")
plt.plot(X, X_with_bias @ theta_best, color='red', label="Best-Fit Line")
plt.title("Linear Regression: Income vs House Value")
plt.xlabel("Median Income ($10k units)")
plt.ylabel("Median House Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eda_linear_regression_income_vs_value.png")
plt.show()
print("Saved plot as 'eda_linear_regression_income_vs_value.png'")


# ---- K-Means Clustering from Scratch ----
print("Running K-Means Clustering: Grouping houses based on location...")


# A helper function to measure distance between 2 points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# K-Means Algorithm
def k_means(X, k=4, max_iters=100):
    # Randomly pick 'k' starting points (called centroids)
    centroids = X[sample(range(len(X)), k)]
    
    for step in range(max_iters):
        # Step 1: Group every point with its closest centroid
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        
        # Step 2: Move each centroid to the average of its cluster
        new_centroids = []
        for cluster in clusters:
            cluster = np.array(cluster)
            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # If a cluster ends up empty, reinitialize it randomly
                new_centroids.append(X[sample(range(len(X)), 1)[0]])
        
        new_centroids = np.array(new_centroids)
        
        # Step 3: Stop if centroids aren't changing much anymore
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids, clusters

# Let’s use latitude and longitude for this clustering
geo_data = df[['latitude', 'longitude']].values
centroids, clusters = k_means(geo_data, k=4)

# Let's visualize our clusters!
colors = ['red', 'blue', 'green', 'purple']
plt.figure(figsize=(10,8))
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:,1], cluster[:,0], label=f'Cluster {i+1}', alpha=0.6, s=20, color=colors[i])
plt.scatter([c[1] for c in centroids], [c[0] for c in centroids], color='black', marker='X', s=100, label='Centroids')
plt.title("K-Means Clustering of Houses Based on Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eda_kmeans_geo_clusters.png")
plt.show()
print("Saved plot as 'eda_kmeans_geo_clusters.png'")



# -------- Naive Bayes from Scratch --------
print("Running Naive Bayes classification...")

# For Naive Bayes, we bin continuous features
df['income_bin'] = pd.cut(df['median_income'], bins=5, labels=False)
df['age_bin'] = pd.cut(df['housing_median_age'], bins=5, labels=False)
price_median = df['median_house_value'].median()
df['price_class'] = (df['median_house_value'] > price_median).astype(int)

# Prepare features
features_nb = df[['income_bin', 'age_bin']].astype(int).values
labels_nb = df['price_class'].values

# Split
split_index = int(0.8 * len(features_nb))
X_train, y_train = features_nb[:split_index], labels_nb[:split_index]
X_test, y_test = features_nb[split_index:], labels_nb[split_index:]

# Calculate prior probabilities
from collections import Counter

priors = Counter(y_train)
for cls in priors:
    priors[cls] /= len(y_train)

# Calculate likelihoods
likelihoods = {}
for cls in np.unique(y_train):
    class_data = X_train[y_train == cls]
    likelihoods[cls] = {}
    for col in range(X_train.shape[1]):
        values, counts = np.unique(class_data[:, col], return_counts=True)
        total = len(class_data)
        likelihoods[cls][col] = {v: c / total for v, c in zip(values, counts)}

# Prediction function
def predict(x):
    posteriors = {}
    for cls in priors:
        prob = np.log(priors[cls])
        for i, val in enumerate(x):
            prob += np.log(likelihoods[cls][i].get(val, 1e-6))  # Laplace smoothing
        posteriors[cls] = prob
    return max(posteriors, key=posteriors.get)

# Predict on test set
y_pred = np.array([predict(x) for x in X_test])

# Evaluate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Median", "Above Median"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.savefig("eda_naive_bayes_confusion_matrix.png")
plt.show()
print("Saved confusion matrix to 'eda_naive_bayes_confusion_matrix.png'")



# ---- Logistic Regression from Scratch ----
# print("Running Logistic Regression: Predicting house value class based on income and age...")

# # Use the same binned features as in Naive Bayes
# X = df[['income_bin', 'age_bin']].astype(int).values
# y = df['price_class'].values

# # Train-test split (80/20)
# split_index = int(0.8 * len(X))
# X_train, y_train = X[:split_index], y[:split_index]
# X_test, y_test = X[split_index:], y[split_index:]

# # Add intercept term
# X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
# X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# # Sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # Loss function (binary cross-entropy)
# def compute_loss(y, h):
#     epsilon = 1e-8
#     return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

# # Gradient Descent
# def train_logistic_regression(X, y, lr=0.1, epochs=1000):
#     theta = np.zeros(X.shape[1])
#     for i in range(epochs):
#         z = np.dot(X, theta)
#         h = sigmoid(z)
#         gradient = np.dot(X.T, (h - y)) / y.size
#         theta -= lr * gradient
        
#         if i % 100 == 0:
#             loss = compute_loss(y, h)
#             print(f"Epoch {i}, Loss: {loss:.4f}")
#     return theta

# # Train the model
# theta = train_logistic_regression(X_train_bias, y_train)

# # Predict function
# def predict(X, theta, threshold=0.5):
#     return sigmoid(np.dot(X, theta)) >= threshold

# # Make predictions
# y_pred = predict(X_test_bias, theta).astype(int)

# # Evaluate accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# # Confusion Matrix
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Median", "Above Median"])
# disp.plot(cmap="Purples")
# plt.title("Confusion Matrix - Logistic Regression")
# plt.savefig("eda_logistic_regression_confusion_matrix.png")
# plt.show()
# print("Saved plot as 'eda_logistic_regression_confusion_matrix.png'")




