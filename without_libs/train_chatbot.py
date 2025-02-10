import json
import numpy as np
import joblib


def load_intents(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def tokenize(text):
    text = text.lower()
    tokens = []
    current_word = []
    for char in text:
        if "a" <= char <= "z":
            current_word.append(char)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
    if current_word:
        tokens.append("".join(current_word))
    return tokens


def build_vocabulary(documents):
    vocab_set = set()
    for tokens in documents:
        for token in tokens:
            vocab_set.add(token)
    return list(vocab_set)


def compute_tf(doc_tokens, vocab):  # formula: (count of t in d / total tokens in d)
    total_tokens = len(doc_tokens)
    if total_tokens == 0:
        return [0.0] * len(vocab)

    token_count = {}
    for token in doc_tokens:
        token_count[token] = token_count.get(token, 0) + 1

    tf_vector = []
    for word in vocab:
        count_w = token_count.get(word, 0)
        tf_vector.append(count_w / total_tokens)
    return tf_vector


def compute_idf(all_docs, vocab):  # formula: log(N of docs / (1 + docs with term t))
    import math

    N = len(all_docs)
    idf_values = []
    for word in vocab:
        doc_count = 0
        for tokens in all_docs:
            if word in tokens:
                doc_count += 1
        idf = math.log(N / (1 + doc_count))
        idf_values.append(idf)
    return idf_values


def compute_tfidf(all_docs_tokens, vocab, idf_values):
    tfidf_matrix = []
    for tokens in all_docs_tokens:
        tf_vector = compute_tf(tokens, vocab)
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_values)]
        tfidf_matrix.append(tfidf_vector)
    return tfidf_matrix


class MultinomialNaiveBayes:

    def fit(
        self, tfidf_matrix, labels, alpha=1.0
    ):  # tfidf_matrix: tfidf for each term in each document
        self.classes_ = list(set(labels))
        self.class_count_ = {c: 0 for c in self.classes_}  # N of docs in each class

        """ docs_by_class

            Separate documents by class  {
                'label': [
                    [0.321, 0.123, ...],      - tfidf_vector1
                    [tfidf_vector2],
                    [tfidf_vector3],
                    ...
                ]
            }
        """
        docs_by_class = {c: [] for c in self.classes_}

        for features, label in zip(tfidf_matrix, labels):
            docs_by_class[label].append(features)
            self.class_count_[label] += 1

        # Compute class priors: P(c) = #docs_in_class / total_docs
        self.priors_ = {}  # prior probability of each class
        total_docs = len(labels)
        for c in self.classes_:
            self.priors_[c] = self.class_count_[c] / total_docs

        self.word_likelihoods_ = {}
        vocab_size = len(tfidf_matrix[0]) if tfidf_matrix else 0

        for c in self.classes_:
            combined_tfidf = np.sum(docs_by_class[c], axis=0)

            total_tfidf_in_class = np.sum(combined_tfidf)

            self.word_likelihoods_[c] = []
            for i in range(vocab_size):
                numerator = combined_tfidf[i] + alpha
                denominator = total_tfidf_in_class + alpha * vocab_size

                self.word_likelihoods_[c].append(numerator / denominator)

    def predict(self, tfidf_vector):
        predictions = []
        for doc_vec in tfidf_vector:
            best_class = None
            best_score = float("-inf")

            for c in self.classes_:
                import math

                score = math.log(self.priors_[c] + 1e-9)

                for i, val in enumerate(doc_vec):
                    score += val * math.log(self.word_likelihoods_[c][i] + 1e-9)

                if score > best_score:
                    best_score = score
                    best_class = c

            predictions.append(best_class)
        return predictions

    def predict_class(self, vocab, idf_vals, text):
        tokens = tokenize(text)
        tf_vector = compute_tf(tokens, vocab)
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vals)]
        prediction = self.predict([tfidf_vector])
        return prediction[0]


def train_model(intents_file):
    # Load data
    data = load_intents(intents_file)

    patterns = []  # array of all 'message' strings
    labels = (
        []
    )  # labels for each 'message' string ['a', 'a', 'a', 'b', 'b', ...] (N messages per tag = N repetetive labels)
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            labels.append(tag)

    # array of arrays of 'message' strings tokenized [['How', 'are', 'you'], [...], ...]    ]
    tokenized_docs = [tokenize(p) for p in patterns]

    # Vocabulary – all unique words in all 'message' strings over all tags ['activities', 'games', ...]
    vocab = build_vocabulary(tokenized_docs)

    # Compute IDF values
    idf_vals = compute_idf(tokenized_docs, vocab)

    # Compute TF-IDF for all documents
    tfidf_matrix = compute_tfidf(tokenized_docs, vocab, idf_vals)

    # Train Naive Bayes
    nb = MultinomialNaiveBayes()
    nb.fit(tfidf_matrix, labels)

    return nb, vocab, idf_vals, data


if __name__ == "__main__":
    model_path = "model.pkl"
    nb, vocab, idf_vals, data = train_model("intents.json")

    model_data = {"model": nb, "vocab": vocab, "idf_vals": idf_vals, "data": data}
    joblib.dump(model_data, model_path)

    print(f"Model saved to {model_path}")
