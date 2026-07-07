import pandas as pd
import numpy as np
import re
import math
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
from sklearn.inspection import PartialDependenceDisplay

# ==========================================
# 1. HELPER: ENTROPY CALCULATION
# ==========================================
def calculate_entropy(text):
    if not text: return 0
    entropy = 0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

# ==========================================
# 2. FEATURE EXTRACTION (WITH DEFENSIVE PARSING)
# ==========================================
def extract_features(url):
    features = {}
    url_with_scheme = url if "://" in url else "http://" + url
    
    domain = ""
    path = ""
    
    try:
        parsed = urlparse(url_with_scheme)
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path
    except (ValueError, Exception):
        pass

    clean_url = str(url).lower().replace("https://", "").replace("http://", "").replace("www.", "")
    
    features['url_length'] = len(clean_url)
    features['domain_length'] = len(domain)
    features['count_at'] = clean_url.count('@')
    features['count_dash'] = clean_url.count('-')
    features['count_dot'] = clean_url.count('.')
    features['count_dir'] = clean_url.count('/')
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', clean_url) else 0
    
    suspicious_words = ['login', 'secure', 'account', 'update', 'verify', 'bank', 'paypal', 'auth', 'admin']
    features['suspicious_word_count'] = sum(1 for word in suspicious_words if word in clean_url)
    features['entropy'] = calculate_entropy(clean_url)
    
    digits = sum(c.isdigit() for c in clean_url)
    features['digit_ratio'] = digits / len(clean_url) if len(clean_url) > 0 else 0

    features['domain_dot_count'] = domain.count('.')
    features['path_depth'] = path.count('/')
    
    dangerous_ext = ['.php', '.exe', '.sh', '.bin', '.js', '.zip']
    features['has_dangerous_ext'] = 1 if any(ext in path.lower() for ext in dangerous_ext) else 0
    
    risky_tlds = ['.xyz', '.top', '.live', '.buzz', '.gq', '.tk']
    features['is_risky_tld'] = 1 if any(clean_url.endswith(tld) for tld in risky_tlds) else 0

    return features

if __name__ == "__main__":
    # ==========================================
    # 3. DATA LOADING & CLEANING
    # ==========================================
    print(">>> 1. Loading the dataset...")
    df = pd.read_csv('malicious_urls.csv') 
    if 'type' in df.columns:
        df.rename(columns={'type': 'label'}, inplace=True)

    print(f"Original Data Shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"Cleaned Data Shape: {df.shape}")

    # ==========================================
    # 4. DATA ANALYSIS (EDA)
    # ==========================================
    print("\n>>> 2. Performing Data Analysis (Close the window to continue)...")
    class_counts = df['label'].value_counts()
    print("Class Distribution:\n", class_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
    plt.title('Threat Type Distribution')
    plt.show() # Live display, script pauses until closed

    # ==========================================
    # 5. FEATURE EXTRACTION & SCALING
    # ==========================================
    print("\n>>> 3. Extracting features (Processing 100k sample for speed)...")
    df_sampled = df.sample(n=100000, random_state=42) 
    
    extracted = df_sampled['url'].apply(lambda x: extract_features(str(x)))
    X = pd.DataFrame(extracted.tolist())

    le = LabelEncoder()
    y = le.fit_transform(df_sampled['label'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 6. ENSEMBLE TRAINING
    # ==========================================
    print("\n>>> 4. Training Ensemble Models...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"   - RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    print(f"   - LR Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}")

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    print(f"   - GB Accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}")

    # ==========================================
    # 7. SAVE ARTIFACTS
    # ==========================================
    print("\n>>> 5. Saving all models and artifacts offline...")
    joblib.dump(rf, 'model_rf.pkl')
    joblib.dump(lr, 'model_lr.pkl')
    joblib.dump(gb, 'model_gb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')
    print(">>> SUCCESS: All models are ready for offline use.")

    # ==========================================
    # 8. PRESENTATION VISUALIZATIONS (LIVE VIEW)
    # ==========================================
    feature_names = list(X.columns)
    print("\n>>> 6. Generating Presentation Visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- VISUAL 1: RANDOM FOREST ---
    print("   -> Displaying Random Forest visuals (Close the window to continue)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    plot_tree(rf.estimators_[0], max_depth=3, feature_names=feature_names, 
              class_names=[str(c) for c in le.classes_], filled=True, rounded=True, 
              ax=axes[0], fontsize=8)
    axes[0].set_title('Inside the Forest: Structure of a Single Decision Tree\n(Node Splitting Logic)', fontsize=14, fontweight='bold')

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-10:] 
    axes[1].barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
    axes[1].set_yticks(range(len(indices)), [feature_names[i] for i in indices])
    axes[1].set_title('Forest Distribution: Top 10 Most Important Features', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Relative Importance')

    plt.tight_layout()
    plt.show() 

   # --- VISUAL 2: LOGISTIC REGRESSION ---
    print("   -> Displaying Logistic Regression visuals (Close the window to continue)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: Feature Coefficients (Weights)
    # lr.coef_ is a 2D array in multi-class. We visualize the weights for the first class (usually 'benign' or the first threat)
    coefs = lr.coef_[0]
    top_indices = np.argsort(np.abs(coefs))[-10:] 
    colors = ['#e74c3c' if c < 0 else '#3498db' for c in coefs[top_indices]]

    axes[0].barh(range(len(top_indices)), coefs[top_indices], color=colors)
    axes[0].set_yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    axes[0].set_title(f'Linear Weights: Impact on "{le.classes_[0]}"', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value')

    # Right (The Surprise): The Sigmoid Probability Mapping
    # FIX: Conditionally handle multi-class vs binary data shapes
    if len(le.classes_) > 2:
        # Multi-class: Grab the highest confidence score and highest probability per URL
        z = np.max(lr.decision_function(X_test), axis=1)
        probs = np.max(lr.predict_proba(X_test), axis=1)
    else:
        # Binary
        z = lr.decision_function(X_test)
        probs = lr.predict_proba(X_test)[:, 1]

    # Sort for a smooth curve
    sorted_indices = np.argsort(z)
    z_sorted = z[sorted_indices]
    probs_sorted = probs[sorted_indices]

    axes[1].plot(z_sorted, probs_sorted, color='#9b59b6', lw=3)
    axes[1].scatter(z, probs, color='black', alpha=0.1, s=10) 
    axes[1].set_title('The Sigmoid Squash: Mapping Raw Scores to Probabilities', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Max Linear Combination (Z)')
    axes[1].set_ylabel('Engine Confidence Probability')

    plt.tight_layout()
    plt.show()

    # --- VISUAL 3: GRADIENT BOOSTING ---
    print("   -> Displaying Gradient Boosting visuals (Close the window to finish)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: Training Deviance (Loss Curve over iterations)
    train_score = np.zeros((gb.n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(gb.staged_predict(X_train)):
        train_score[i] = accuracy_score(y_train, y_pred)

    axes[0].plot(np.arange(gb.n_estimators) + 1, train_score, 'b-', linewidth=2)
    axes[0].set_title('The Additive Process: Accuracy Gained per Boosting Stage', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Boosting Iterations (Trees)')
    axes[0].set_ylabel('Training Accuracy')

    # Right (The Surprise): Partial Dependence Plot (PDP)
    top_2_features = [feature_names[i] for i in np.argsort(gb.feature_importances_)[-2:]]
    
    # FIX: Specify the target class for multi-class partial dependence
    target_class_idx = 0  # Change this to 1, 2, or 3 to see the plot for other threat types!
    target_class_name = le.classes_[target_class_idx]

    PartialDependenceDisplay.from_estimator(
        gb, X_train, features=top_2_features, feature_names=feature_names,
        target=target_class_idx, ax=axes[1], grid_resolution=50
    )
    axes[1].set_title(f'Partial Dependence: How {top_2_features[0]} & {top_2_features[1]}\nImpact "{target_class_name}" Probability', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()