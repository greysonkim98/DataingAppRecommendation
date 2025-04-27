import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# -----------------------------
# Check the sentences similarity  
# -----------------------------
def cosine_similarity(X, Y):
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    sim = np.dot(X, Y.T) / (X_norm * Y_norm.T + 1e-10)
    return sim
# -----------------------------
# Loss
# -----------------------------
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(m), y_true] + 1e-10)
    loss = np.sum(log_likelihood) / m
    return loss

# -----------------------------
# Softmax for classification
# -----------------------------
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-10)

def train_softmax(X, y, num_classes, lr, epochs):
    np.random.seed(42)
    n_samples, n_features = X.shape
    W = np.random.randn(n_features, num_classes) * 0.01
    b = np.zeros((1, num_classes))
    losses = []

    for epoch in range(epochs):
        logits = np.dot(X, W) + b
        probs = softmax(logits)

        loss = cross_entropy(y, probs)
        losses.append(loss)

        # gradient decent
        probs_copy = probs.copy()
        probs_copy[np.arange(n_samples), y] -= 1
        dW = np.dot(X.T, probs_copy) / n_samples
        db = np.sum(probs_copy, axis=0, keepdims=True) / n_samples

        W -= lr * dW
        b -= lr * db

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    return W, b, losses

# Return the biggest probabilities
def predict(X, W, b):
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

# -----------------------------
# Evaluation
# -----------------------------
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    accuracy = np.sum(TP) / np.sum(cm)
    precision = np.mean(TP / (TP + FP + 1e-10))
    recall = np.mean(TP / (TP + FN + 1e-10))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1

# Plot confusion matrix in heatmap
def plot_confusion_matrix(cm, class_names):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
# -----------------------------
# Feature Scoring
# -----------------------------
def load_or_score_reviews(app, features, model, base_dir="./"):
    scored_path = os.path.join(base_dir, f"{app}_scored_reviews.csv")
    if os.path.exists(scored_path):
        print(f"Loading {scored_path}")
        return pd.read_csv(scored_path)
    else:
        texts = pd.read_csv(os.path.join(base_dir, f"{app}_balanced_reviews.csv"))['content'].astype(str).tolist()
        embeddings = np.load(os.path.join(base_dir, f"{app}_embeddings.npy"))
        feature_embeds = model.encode(list(features.values()), show_progress_bar=False)
        cosine_scores = cosine_similarity(embeddings, feature_embeds)
        score_df = pd.DataFrame(cosine_scores, columns=features.keys())
        score_df = score_df.apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-10) * 10, axis=0)
        result_df = pd.concat([pd.read_csv(os.path.join(base_dir, f"{app}_balanced_reviews.csv")).reset_index(drop=True), score_df], axis=1)
        result_df.to_csv(scored_path, index=False)
        print(f"✅ Saved {scored_path}")
        return result_df

# -----------------------------
# Main Functions
# -----------------------------
def main():
    apps = ["bumble", "tinder", "hinge"]
    # Define embeding sentences
    features = {
        "Committed Relationship": "I'm looking for a serious, long-term relationship, not just casual chats or hookups.",
        "Initiating Conversation": "I don’t mind starting conversations first — I'm pretty proactive when I’m interested in someone.",
        "Willingness to Pay for Love": "I’m okay with paying for premium features if it helps me find someone meaningful.",
        "Distance Flexibility": "Distance doesn’t matter much to me as long as I find someone I truly connect with.",
        "Clear Ideal Type": "I have a pretty clear picture of what I want in a partner and won’t settle for less.",
        "Importance of Appearance": "Physical appearance plays a big role for me when choosing who to date.",
        "Customer Support Experience": "If I run into issues, I expect the app to have responsive and helpful customer support."
    }
    all_data = []
    labels = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load dataset
    for idx, app in enumerate(apps):
        df = load_or_score_reviews(app, features, model)
        all_data.append(df[list(features.keys())].values)
        labels.append(np.full((len(df),), idx))
    # Balance each dataset
    min_samples = min([len(x) for x in all_data])
    print(f"Balancing to {min_samples} samples each")

    balanced_data = []
    balanced_labels = []
    for i in range(len(all_data)):
        idx = np.random.choice(len(all_data[i]), min_samples, replace=False)
        balanced_data.append(all_data[i][idx])
        balanced_labels.append(labels[i][idx])

    # Normalize Data 
    X = np.vstack(balanced_data)
    y = np.hstack(balanced_labels)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-10)

    # Split datasets
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train
    W, b, losses = train_softmax(X_train, y_train, num_classes=3, lr=0.3, epochs=5000)

    # Predict
    y_pred = predict(X_test, W, b)

    # Evaluate
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred, num_classes=3)
    print(f"\nAccuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")

    # Confusion
    cm = confusion_matrix(y_test, y_pred, num_classes=3)
    plot_confusion_matrix(cm, class_names=apps)

    # Loss curve
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
