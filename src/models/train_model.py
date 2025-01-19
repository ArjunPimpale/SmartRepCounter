import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from LearningAlgorithm import ClassificationAlgorithms
from sklearn.metrics import accuracy_score

df = pd.read_pickle("../../data/interim/processed_data.pkl")

train_df = df.drop(columns=["category", "candidate", "set","SetDuration"])

X = train_df.drop(columns=["label"])
y = train_df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# plot to show the distribution of labels in the train and test sets
# Get unique labels
unique_labels = np.unique(y)

# Calculate counts for train and test sets
train_counts = [np.sum(y_train == label) for label in unique_labels]
test_counts = [np.sum(y_test == label) for label in unique_labels]

# Set up the bar plot
x = np.arange(len(unique_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, train_counts, width, label="Train")
rects2 = ax.bar(x + width / 2, test_counts, width, label="Test")

# Customize the plot
ax.set_ylabel("Count")
ax.set_title("Label Distribution in Train and Test Sets")
ax.set_xticks(x)
ax.set_xticklabels([f"Label {label}" for label in unique_labels])
ax.legend()


# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acc_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in train_df.columns if "_temp_" in f]
freq_features = [f for f in train_df.columns if "_freq" in f or "_pse" in f]
cluster_features = ["clusters"]

print("Features", len(basic_features))
print("Features", len(square_features))
print("Features", len(pca_features))
print("Features", len(time_features))
print("Features", len(freq_features))
print("Features", len(cluster_features))

feature_1 = basic_features
feature_2 = list(set(basic_features + square_features + pca_features))
feature_3 = list(set(feature_2 + time_features))
feature_4 = list(set(feature_3 + freq_features + cluster_features))

# Using decision tree of forward selection to select the best features

learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)
X_train.columns
selected_features = ['acc_y_freq_0.0_Hz_ws_14',
 'gyro_r_freq_0.0_Hz_ws_14',
 'acc_y_temp_std_ws_5',
 'acc_y_temp_mean_ws_5',
 'acc_z_freq_0.0_Hz_ws_14',
 'gyro_y',
 'acc_z_pse',
 'gyro_r_freq_1.786_Hz_ws_14',
 'gyro_y_freq_1.429_Hz_ws_14',
 'gyro_z_freq_0.357_Hz_ws_14']



# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(1,max_features+1,1) , ordered_scores)
# plt.xlabel("Features")
# plt.ylabel("Accuracy")
# plt.xticks(np.arange(1,max_features+1,1))
# plt.show()


possible_feature_sets = [
    feature_1,
    feature_2,
    feature_3,
    feature_4,
    selected_features
]
feature_names = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "selected_features"
]

iterations = 1
score_df = pd.DataFrame()
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
