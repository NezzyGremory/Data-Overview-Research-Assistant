"""Optional, CPU-first neural estimators with lazy framework imports."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), epochs=30, batch_size=256, learning_rate=0.001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        features = np.asarray(X, dtype=np.float32)
        self.classes_, labels = np.unique(y, return_inverse=True)
        torch.manual_seed(self.random_state)
        layers = []
        width = features.shape[1]
        for units in self.hidden_layer_sizes:
            layers.extend((nn.Linear(width, units), nn.ReLU()))
            width = units
        layers.append(nn.Linear(width, len(self.classes_)))
        self.model_ = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        x_tensor = torch.from_numpy(np.ascontiguousarray(features))
        y_tensor = torch.as_tensor(labels, dtype=torch.long)
        self.model_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(x_tensor))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(self.model_(x_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimizer.step()
        self.n_features_in_ = features.shape[1]
        return self

    def predict_proba(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            values = torch.as_tensor(np.asarray(X, dtype=np.float32))
            return torch.softmax(self.model_(values), dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), epochs=30, batch_size=256, learning_rate=0.001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        torch.manual_seed(self.random_state)
        layers = []
        width = features.shape[1]
        for units in self.hidden_layer_sizes:
            layers.extend((nn.Linear(width, units), nn.ReLU()))
            width = units
        layers.append(nn.Linear(width, 1))
        self.model_ = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        x_tensor = torch.from_numpy(np.ascontiguousarray(features))
        y_tensor = torch.from_numpy(np.ascontiguousarray(target))
        self.model_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(x_tensor))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model_(x_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimizer.step()
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            values = torch.as_tensor(np.asarray(X, dtype=np.float32))
            return self.model_(values).cpu().numpy().reshape(-1)


class TensorFlowMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), epochs=30, batch_size=256, learning_rate=0.001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        import tensorflow as tf

        features = np.asarray(X, dtype=np.float32)
        self.classes_, labels = np.unique(y, return_inverse=True)
        tf.keras.utils.set_random_seed(self.random_state)
        layers = [tf.keras.layers.Input(shape=(features.shape[1],))]
        layers.extend(tf.keras.layers.Dense(units, activation="relu") for units in self.hidden_layer_sizes)
        layers.append(tf.keras.layers.Dense(len(self.classes_), activation="softmax"))
        self.model_ = tf.keras.Sequential(layers)
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
        )
        self.model_.fit(features, labels, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.n_features_in_ = features.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TensorFlowMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), epochs=30, batch_size=256, learning_rate=0.001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        import tensorflow as tf

        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32)
        tf.keras.utils.set_random_seed(self.random_state)
        layers = [tf.keras.layers.Input(shape=(features.shape[1],))]
        layers.extend(tf.keras.layers.Dense(units, activation="relu") for units in self.hidden_layer_sizes)
        layers.append(tf.keras.layers.Dense(1))
        self.model_ = tf.keras.Sequential(layers)
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
        )
        self.model_.fit(features, target, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0).reshape(-1)
