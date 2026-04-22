from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = [
    # Spam (dài hơn, tự nhiên hơn)
    "Congratulations! You have won a free prize. Click here to claim now.",
    "Limited offer! Buy cheap products today and save big money.",
    "You have been selected to receive a reward. Act now!",

    # Non-spam
    "Hi, I hope you're doing well. Are we still meeting tomorrow?",
    "Let's review the lesson together before the exam next week.",
    "This is a reminder about your appointment scheduled for tomorrow."
]

labels = ["spam", "spam", "spam", "not spam", "not spam", "not spam"]
# ===== SPLIT DATA =====
train_texts = texts[:4]
train_labels = labels[:4]

test_texts = texts[4:]
test_labels = labels[4:]

# ===== VECTOR =====
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# ===== MODEL =====
model = MultinomialNB()
model.fit(X_train, train_labels)

# ===== EVALUATION =====
predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)

print("Accuracy:", accuracy)

# ===== TEST TABLE =====
test_samples = [
    "Congratulations! You have won a free iPhone. Click here now!",
    "Hi, can we discuss the assignment tomorrow?",
    "Limited time offer! Buy now and get 50% discount!",
    "Please submit your homework before Friday."
]

vectors = vectorizer.transform(test_samples)
predictions = model.predict(vectors)

print("\n--- Spam Detection Results ---")
print(f"{'Text':<60} | Prediction")
print("-" * 80)

for text, pred in zip(test_samples, predictions):
   print(f"{text[:40]:<45} | {pred}")
print("\n--- Try Your Own Input ---")
user_input = input("Enter a comment: ")

user_vector = vectorizer.transform([user_input])
result = model.predict(user_vector)

print("Prediction:", result[0])
