import pandas as pd
import numpy as np
import joblib
import math
import re
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score

# ==========================================
# 1. HELPER FUNCTIONS (Must match training exactly)
# ==========================================
def calculate_entropy(text):
    if not text: return 0
    entropy = 0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features(url):
    features = {}
    url_with_scheme = url if "://" in url else "http://" + url
    domain, path = "", ""
    
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

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================
def evaluate_saved_models(csv_file, target_col, prefix, name, sample_size=None):
    print(f"\n{'='*50}")
    print(f" EVALUATING: {name}")
    print(f"{'='*50}")
    
    # 1. Load Artifacts
    try:
        print(f">>> Loading saved models and artifacts (prefix: '{prefix}')...")
        rf = joblib.load(f'{prefix}model_rf.pkl')
        lr = joblib.load(f'{prefix}model_lr.pkl')
        gb = joblib.load(f'{prefix}model_gb.pkl')
        scaler = joblib.load(f'{prefix}scaler.pkl')
        le = joblib.load(f'{prefix}label_encoder.pkl')
    except FileNotFoundError as e:
        print(f"[!] Error: Could not find model files. Did you run the training script first?\nDetails: {e}")
        return

    # 2. Load Dataset
    try:
        df = pd.read_csv(csv_file)
        if target_col not in df.columns and 'type' in df.columns:
            df.rename(columns={'type': 'label'}, inplace=True)
            
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
        df = df.dropna(subset=['url', target_col])
    except FileNotFoundError:
        print(f"[!] Error: Dataset '{csv_file}' not found.")
        return

    # 3. Extract & Process Features
    print(">>> Extracting features from URLs (this may take a moment)...")
    extracted = df['url'].apply(lambda x: extract_features(str(x)))
    X = pd.DataFrame(extracted.tolist())
    
    X_scaled = scaler.transform(X)
    y_true = le.transform(df[target_col])

    # 4. Individual Predictions & Accuracies
    print("\n>>> Calculating Individual Model Accuracies...")
    rf_preds = rf.predict(X_scaled)
    lr_preds = lr.predict(X_scaled)
    gb_preds = gb.predict(X_scaled)

    print(f"   - Random Forest (RF):      {accuracy_score(y_true, rf_preds) * 100:.2f}%")
    print(f"   - Logistic Regression (LR): {accuracy_score(y_true, lr_preds) * 100:.2f}%")
    print(f"   - Gradient Boosting (GB):   {accuracy_score(y_true, gb_preds) * 100:.2f}%")

    # 5. Soft Voting Accuracy (Manual Ensemble)
    print("\n>>> Calculating Soft Voting Overall Accuracy...")
    # Get probability predictions from all 3 models
    rf_probs = rf.predict_proba(X_scaled)
    lr_probs = lr.predict_proba(X_scaled)
    gb_probs = gb.predict_proba(X_scaled)

    # Average the probabilities (Soft Voting mechanism)
    soft_voting_probs = (rf_probs + lr_probs + gb_probs) / 3
    
    # Pick the class with the highest average probability
    soft_voting_preds = np.argmax(soft_voting_probs, axis=1)
    
    soft_acc = accuracy_score(y_true, soft_voting_preds)
    print(f"   => OVERALL SOFT VOTING ACCURACY: {soft_acc * 100:.2f}%\n")


if __name__ == "__main__":
    # Test Code 1 (Phishing Dataset)
    # File prefix used in Code 1 was "phish_"
    evaluate_saved_models(
        csv_file='dataset_phishing.csv', 
        target_col='status', 
        prefix='phish_', 
        name="Code 1 (dataset_phishing.csv)"
    )

    # Test Code 2 (Malicious URLs Dataset)
    # File prefix used in Code 2 was empty ""
    # We use a 20,000 sample size here to make the test evaluation quick
    evaluate_saved_models(
        csv_file='malicious_urls.csv', 
        target_col='label', 
        prefix='', 
        name="Code 2 (malicious_urls.csv)",
        sample_size=20000 
    )