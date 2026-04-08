import math
from collections import Counter

# --- DECISION TREE LOGIC (ID3 & C4.5) ---

def calculate_entropy(data):
    if not data: return 0
    labels = [row[-1] for row in data]
    counts = Counter(labels)
    probs = [count / len(labels) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def get_info_gain(data, feature_idx):
    total_entropy = calculate_entropy(data)
    values = set(row[feature_idx] for row in data)
    weighted_entropy = 0
    for val in values:
        subset = [row for row in data if row[feature_idx] == val]
        weighted_entropy += (len(subset) / len(data)) * calculate_entropy(subset)
    return total_entropy - weighted_entropy

def get_gain_ratio(data, feature_idx):
    ig = get_info_gain(data, feature_idx)
    feature_values = [row[feature_idx] for row in data]
    counts = Counter(feature_values)
    split_info = -sum((c/len(data)) * math.log2(c/len(data)) for c in counts.values() if c > 0)
    return ig / split_info if split_info != 0 else 0

# --- NAIVE BAYES CLASSIFIER ---

class NaiveBayesScratch:
    def train(self, data, labels):
        self.priors = Counter([row[-1] for row in data])
        self.total = len(data)
        self.likelihoods = {} 
        self.unique_labels = labels
        
        for i in range(len(data[0]) - 1):
            for row in data:
                key = (i, row[i], row[-1])
                self.likelihoods[key] = self.likelihoods.get(key, 0) + 1

    def predict(self, features):
        probs = {}
        for label in self.unique_labels:
            prior = self.priors[label] / self.total
            likelihood = 1.0
            for i, val in enumerate(features):
                count = self.likelihoods.get((i, val, label), 0)
                # Laplace Smoothing
                likelihood *= (count + 1) / (self.priors[label] + len(self.unique_labels))
            probs[label] = prior * likelihood
        return max(probs, key=probs.get)