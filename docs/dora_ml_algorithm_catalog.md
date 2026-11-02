# Katalog algoritma ML

Inventaris `app/ml/registry.py` pada 2026-11-02 berisi **49 spesifikasi**: 18 klasifikasi, 20 regresi, 7 clustering, 4 anomaly detection. Pada snapshot `.venv` lokal, pemeriksaan dependency registry menyatakan 39 tersedia dan 10 model dengan PyTorch, TensorFlow, XGBoost, LightGBM atau CatBoost perlu paket opsional. Angka ini adalah *registrasi dan ketersediaan import*, bukan 39 hasil benchmark yang lulus.

| Task | Estimator terdaftar dengan dependency dasar | Opsi yang memerlukan dependency lain |
| --- | --- | --- |
| Klasifikasi | Logistic Regression, SGD, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, SVM, KNN, Gaussian NB, LDA, MLP sklearn | MLP PyTorch/TensorFlow, XGBoost, LightGBM, CatBoost |
| Regresi | Linear, Ridge, Lasso, Elastic Net, SGD, Huber, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, SVR, KNN, MLP sklearn | MLP PyTorch/TensorFlow, XGBoost, LightGBM, CatBoost |
| Clustering | K-Means, MiniBatch K-Means, Agglomerative, DBSCAN, Gaussian Mixture, Birch, Spectral | HDBSCAN dan Mean Shift tidak menjadi entri registry utama. |
| Anomali | Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope | Evaluasi supervised memerlukan label ground truth yang belum menjadi input task ini. |
| Forecasting | Naive dan Seasonal Naive di `evaluation.py` | ARIMA, Exponential Smoothing, SARIMAX memerlukan `statsmodels`; metode ini berada di luar registry 49. |

`ModelRegistry.instantiate` membangun estimator; `CandidateModelSelector` membatasi kandidat sesuai task/ukuran; `MLEvaluator` menjalankan kandidat di dataset pengguna. Dukungan task saat ini: klasifikasi biner/multikelas, regresi, clustering, anomaly detection, forecasting. Hasil untuk metode tertentu dapat gagal bila data, asumsi, paket, atau sumber daya tidak sesuai dan kegagalan harus dilaporkan.

**Belum tersedia sebagai task terpadu:** semi-supervised, reinforcement learning, reduksi dimensi yang dievaluasi tersendiri, CNN/RNN/LSTM/GRU/Transformer, voting/stacking otomatis di ML Intelligence, serta pencarian hiperparameter yang terhubung UI dan tracking. Node Editor memiliki sebagian node ensemble/preprocessing terpisah; itu tidak sama dengan dukungan eksperimen ML Intelligence lengkap.
