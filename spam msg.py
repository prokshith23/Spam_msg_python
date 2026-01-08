
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv(
    r"C:\Downloads\archive (4)\spam_sms.csv",   # change path if needed
    encoding="latin-1"
)

# Rename columns (FIXED FOR THIS DATASET)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

print("Dataset Loaded Successfully")
print(df.head())
print("\nClass Distribution:")
print(df['label'].value_counts())

# -----------------------------
# 3. Text Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Encode labels
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'],
    df['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_encoded']
)

# -----------------------------
# 5. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6. Models
# -----------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}

# -----------------------------
# 7. Train & Evaluate
# -----------------------------
for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam']
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# -----------------------------
# 8. Test on New SMS
# -----------------------------
sample_sms = [
    "Congratulations you won free recharge",
    "Hey are we meeting tomorrow",
    "Urgent call now to claim prize"
]

sample_vec = vectorizer.transform(sample_sms)
predictions = models["Naive Bayes"].predict(sample_vec)

print("\nSample Predictions:")
for sms, pred in zip(sample_sms, predictions):
    print(f"{sms} --> {'Spam' if pred == 1 else 'Ham'}")


