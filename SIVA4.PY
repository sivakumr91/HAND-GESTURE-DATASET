import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Dogs vs Cats Dataset simulation (HOG-like flattened features for SVM) [web:25][web:28][web:29]
np.random.seed(42)
n_samples_per_class = 200
n_features = 100

cat_features = np.random.normal(0.4, 0.2, (n_samples_per_class, n_features))
dog_features = np.random.normal(0.6, 0.2, (n_samples_per_class, n_features))
X = np.vstack([cat_features, dog_features])
y = np.array(['cat'] * n_samples_per_class + ['dog'] * n_samples_per_class)

df = pd.DataFrame(X)
df['label'] = y

print("Dataset preview:")
print(df.head())
print("\nDataset shape:", df.shape)
print("Class distribution:")
print(df['label'].value_counts())

# Features and target
X_features = df.drop('label', axis=1)
y_labels = df['label'].map({'cat': 0, 'dog': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, 
                                                    test_size=0.3, random_state=42, stratify=y_labels)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nExplained variance by first 2 PCs: {pca.explained_variance_ratio_.sum():.3f}")

# Train SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

# Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print("\nSVM Model Performance:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Cat', 'Dog']))

# Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# PCA scatter plot (fixed - no DecisionBoundariesDisplay)
plt.subplot(1, 2, 2)
colors = ['red' if l == 0 else 'blue' for l in y_train]
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, alpha=0.7, s=50)
# SVM decision boundary approximation in PCA space
xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min()-1, X_train_pca[:, 0].max()+1, 50),
                     np.linspace(X_train_pca[:, 1].min()-1, X_train_pca[:, 1].max()+1, 50))
Z = svm_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.colorbar(label='Predicted Class')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('SVM Decision Boundary (PCA Space)')

plt.tight_layout()
plt.show()

# Feature importance (SVM coefficients for linear kernel equivalent)
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
feature_importance = np.abs(svm_linear.coef_[0])
top_features = np.argsort(feature_importance)[-10:]
print(f"\nTop 10 most important features: {top_features}")

# Example prediction
new_image = np.random.normal(0.5, 0.2, n_features)
new_scaled = scaler.transform([new_image])
pred = svm_model.predict(new_scaled)[0]
prob = svm_model.predict_proba(new_scaled)[0]
print(f"\nExample prediction - New image: {'Dog' if pred == 1 else 'Cat'}")
print(f"Probabilities: Cat={prob[0]:.3f}, Dog={prob[1]:.3f}")

print("\n✅ Complete SVM Cats vs Dogs classifier ready!")
print("For real Kaggle images: Replace feature extraction with HOG + OpenCV")
