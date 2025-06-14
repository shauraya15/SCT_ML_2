import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Loading the dataset — this one has customer info
df = pd.read_csv("Mall_Customers.csv")

# We'll focus on income and spending behavior for this
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Trying different values of k to figure out what's optimal (using the elbow method)
inertia_vals = []
possible_k = list(range(1, 11))  # I like to keep this separate in case I want to tweak the range later

for num_clusters in possible_k:
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(features)
    inertia_vals.append(model.inertia_)  # Inertia kinda shows how tight the clusters are

# Plotting the elbow curve to visualize where inertia "bends"
plt.figure(figsize=(8, 4))
plt.plot(possible_k, inertia_vals, marker='o', linestyle='--')  # Dashes just for better visibility
plt.title("Elbow Curve to Find Best k")  # Might revisit this title later
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Lower is better)")
plt.grid(True)
plt.savefig("elbow_curve.png")  # Saving just in case I need to put it in a report
plt.show()

# Let's go with 5 clusters for now — seemed like a decent choice from the plot
chosen_k = 5
final_model = KMeans(n_clusters=chosen_k, random_state=42)
cluster_labels = final_model.fit_predict(features)

# Attaching the cluster result back to our main DataFrame
df["Cluster"] = cluster_labels

# Now, visualizing the clusters to see how people are grouped
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",  # Clusters will be colored differently
    palette="tab10",  # Default one I like, visually distinct enough
    s=100  # Slightly bigger points
)

plt.title("Customer Segmentation via K-Means")
plt.savefig("customer_clusters.png")  # Another one for documentation
plt.show()

# Note: Might consider plotting cluster centers too — would help interpret results better
