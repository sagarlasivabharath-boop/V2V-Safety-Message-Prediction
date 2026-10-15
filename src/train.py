import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut

from model import V2VTransformer


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/V2V_Small_Explainable_Dataset.xlsx"

df = pd.read_excel(DATA_PATH)

df.columns = df.columns.str.strip()

print("=" * 60)
print("V2V TRANSFORMER TRAINING PIPELINE")
print("=" * 60)

print("\nDataset size:", len(df))


# --------------------------------------------------
# 2. Load Transformer
# --------------------------------------------------

transformer = V2VTransformer()


# --------------------------------------------------
# 3. Convert V2V messages into embeddings
# --------------------------------------------------

messages = df["message_text"].astype(str).tolist()

print("\nCreating Transformer embeddings...")

embeddings = transformer.encode_messages(messages)

X = embeddings.numpy()

print("Embedding shape:", X.shape)


# --------------------------------------------------
# 4. Prepare criticality labels
# --------------------------------------------------

label_encoder = LabelEncoder()

y_criticality = label_encoder.fit_transform(
    df["criticality"].astype(str)
)

print("\nCriticality classes:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# --------------------------------------------------
# 5. Prepare deadline values
# --------------------------------------------------

deadline_columns = [
    col for col in df.columns
    if "deadline" in col.lower()
]

if not deadline_columns:
    raise ValueError("No deadline column found in the dataset.")

deadline_column = deadline_columns[0]

print("\nUsing deadline column:", deadline_column)

y_deadline = pd.to_numeric(
    df[deadline_column],
    errors="coerce"
).values


# --------------------------------------------------
# 6. Criticality prediction
# --------------------------------------------------

print("\n" + "=" * 60)
print("CRITICALITY CLASSIFICATION")
print("=" * 60)

loo = LeaveOneOut()

actual_criticality = []
predicted_criticality = []

for train_index, test_index in loo.split(X):

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y_criticality[train_index]
    y_test = y_criticality[test_index]

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_test)[0]

    actual_criticality.append(y_test[0])
    predicted_criticality.append(prediction)


# Convert predictions back to names

actual_labels = label_encoder.inverse_transform(
    actual_criticality
)

predicted_labels = label_encoder.inverse_transform(
    predicted_criticality
)


accuracy = accuracy_score(
    actual_criticality,
    predicted_criticality
)

f1 = f1_score(
    actual_criticality,
    predicted_criticality,
    average="weighted"
)

print("\nActual criticality:")
print(actual_labels)

print("\nPredicted criticality:")
print(predicted_labels)

print("\nAccuracy:", round(accuracy, 4))
print("Weighted F1:", round(f1, 4))


# --------------------------------------------------
# 7. Deadline prediction
# --------------------------------------------------

print("\n" + "=" * 60)
print("DEADLINE REGRESSION")
print("=" * 60)

actual_deadlines = []
predicted_deadlines = []

for train_index, test_index in loo.split(X):

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y_deadline[train_index]
    y_test = y_deadline[test_index]

    regressor = Ridge(alpha=1.0)

    regressor.fit(X_train, y_train)

    prediction = regressor.predict(X_test)[0]

    actual_deadlines.append(y_test[0])
    predicted_deadlines.append(prediction)


mae = mean_absolute_error(
    actual_deadlines,
    predicted_deadlines
)

rmse = np.sqrt(
    mean_squared_error(
        actual_deadlines,
        predicted_deadlines
    )
)

print("\nActual deadlines:")
print(np.round(actual_deadlines, 2))

print("\nPredicted deadlines:")
print(np.round(predicted_deadlines, 2))

print("\nMAE:", round(mae, 4), "seconds")
print("RMSE:", round(rmse, 4), "seconds")


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("Number of messages:", len(df))
print("Embedding size:", X.shape[1])
print("Criticality Accuracy:", round(accuracy, 4))
print("Criticality Weighted F1:", round(f1, 4))
print("Deadline MAE:", round(mae, 4), "seconds")
print("Deadline RMSE:", round(rmse, 4), "seconds")

print("\nPrototype experiment completed successfully!")
# --------------------------------------------------
# 9. Generate result graphs
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------- Confusion Matrix ----------


import matplotlib.pyplot as plt
from collections import Counter

# Actual message criticality labels
# Replace y_true with your actual label variable
counts = Counter(y_true)

labels = ["Critical", "High", "Medium", "Low"]
values = [counts.get(label, 0) for label in labels]

plt.figure(figsize=(9, 5))
bars = plt.bar(labels, values)

plt.title("Distribution of V2V Safety Messages")
plt.xlabel("Message Criticality")
plt.ylabel("Number of Messages")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("results/criticality_graph.png", dpi=300)
plt.show()



# ---------- Save Metrics ----------

with open("results/metrics.txt", "w") as file:

    file.write("V2V Transformer Prototype Results\n")
    file.write("=================================\n\n")

    file.write(f"Number of messages: {len(df)}\n")
    file.write(f"Embedding size: {X.shape[1]}\n\n")

    file.write(
        f"Criticality Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Criticality Weighted F1: {f1:.4f}\n"
    )

    file.write(
        f"Deadline MAE: {mae:.4f} seconds\n"
    )

    file.write(
        f"Deadline RMSE: {rmse:.4f} seconds\n"
    )

print("\nResults saved successfully!")

print("results/confusion_matrix.png")
print("results/deadline_prediction.png")
print("results/metrics.txt")