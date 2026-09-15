import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.stats import mode


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income']
df = pd.read_csv(url, names=cols, na_values=' ?', skipinitialspace=True)
df = df.dropna()


X = df.drop('income', axis=1)
y_true = df['income'].map({'<=50K': 0, '>50K': 1})


cat_cols = X.select_dtypes(include='object').columns
for c in cat_cols:
    X[c] = LabelEncoder().fit_transform(X[c])


X_scaled = StandardScaler().fit_transform(X)


km = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = km.fit_predict(X_scaled)


labels_mapped = clusters.copy()
for cluster_id in [0, 1]:
    mask = clusters == cluster_id
    majority_label = mode(y_true[mask], keepdims=True).mode[0]
    labels_mapped[mask] = majority_label


acc = accuracy_score(y_true, labels_mapped)
sil = silhouette_score(X_scaled, clusters)

print(f"Accuracy (after cluster-to-label mapping): {acc:.4f}")
print(f"Silhouette Score: {sil:.4f}")
print(f"Cluster sizes: {pd.Series(clusters).value_counts().to_dict()}")
