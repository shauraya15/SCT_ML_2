import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df = pd.read_csv("Mall_Customers.csv")


features = df[["Annual Income (k$)", "Spending Score (1-100)"]]


inertia_vals = []
possible_k = list(range(1, 11))  

for num_clusters in possible_k:
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(features)
    inertia_vals.append(model.inertia_)  

plt.figure(figsize=(8, 4))
plt.plot(possible_k, inertia_vals, marker='o', linestyle='--') 
plt.title("Elbow Curve to Find Best k") 
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Lower is better)")
plt.grid(True)
plt.savefig("elbow_curve.png")  
plt.show()


chosen_k = 5
final_model = KMeans(n_clusters=chosen_k, random_state=42)
cluster_labels = final_model.fit_predict(features)


df["Cluster"] = cluster_labels


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",  
    palette="tab10", 
    s=100 
)

plt.title("Customer Segmentation via K-Means")
plt.savefig("customer_clusters.png") 
plt.show()


