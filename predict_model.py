import pandas as pd
import numpy as np
import joblib
import sys
import re
import math
from urllib.parse import urlparse

# ==========================================
# 1. FEATURE EXTRACTION (Unified)
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
# 2. PREDICTION AND SOFT VOTING LOGIC
# ==========================================
def get_prediction_details(model, X_scaled, le):
    """Extracts the label, confidence, and unified threat probability."""
    probs = model.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]
    
    # Calculate unified Threat Probability (1.0 - Safe Probability)
    # This maps disparate datasets into a unified Safe vs Threat space
    safe_prob = 0.0
    for i, cls_name in enumerate(le.classes_):
        if str(cls_name).lower() in ['benign', 'legitimate', 'safe']:
            safe_prob += probs[i]
            
    threat_prob = 1.0 - safe_prob
    return pred_label, confidence, threat_prob

def analyze_url(target_url):
    print("\n" + "="*75)
    print(f"ANALYZING: {target_url}")

    # --- A. Load Artifacts for Model Set 1 (General Malicious URLs) ---
    try:
        rf1 = joblib.load('model_rf.pkl') # 80  20
        lr1 = joblib.load('model_lr.pkl')#  70  30
        gb1 = joblib.load('model_gb.pkl')#  60  40
        scaler1 = joblib.load('scaler.pkl')
        le1 = joblib.load('label_encoder.pkl')
        cols1 = joblib.load('feature_names.pkl')
    except FileNotFoundError:
        print("CRITICAL ERROR: Model Set 1 (.pkl) files not found. Run train_model_1.py.")
        return

    # --- B. Load Artifacts for Model Set 2 (Phishing URLs) ---
    try:
        rf2 = joblib.load('phish_model_rf.pkl')
        lr2 = joblib.load('phish_model_lr.pkl')
        gb2 = joblib.load('phish_model_gb.pkl')
        scaler2 = joblib.load('phish_scaler.pkl')
        le2 = joblib.load('phish_label_encoder.pkl')
        cols2 = joblib.load('phish_feature_names.pkl')
    except FileNotFoundError:
        print("CRITICAL ERROR: Model Set 2 (.pkl) files not found. Run train_model_2.py.")
        return

    # --- C. Feature Extraction & Scaling ---
    raw_features = extract_features(target_url)
    
    # Scale for Model 1
    df1 = pd.DataFrame([raw_features]).reindex(columns=cols1, fill_value=0)
    X1 = scaler1.transform(df1)
    
    # Scale for Model 2
    df2 = pd.DataFrame([raw_features]).reindex(columns=cols2, fill_value=0)
    X2 = scaler2.transform(df2)

    # --- D. Get Predictions from all 6 Models ---
    rf1_lbl, rf1_conf, rf1_threat = get_prediction_details(rf1, X1, le1)
    lr1_lbl, lr1_conf, lr1_threat = get_prediction_details(lr1, X1, le1)
    gb1_lbl, gb1_conf, gb1_threat = get_prediction_details(gb1, X1, le1)

    rf2_lbl, rf2_conf, rf2_threat = get_prediction_details(rf2, X2, le2)
    lr2_lbl, lr2_conf, lr2_threat = get_prediction_details(lr2, X2, le2)
    gb2_lbl, gb2_conf, gb2_threat = get_prediction_details(gb2, X2, le2)

    # --- E. Calculate Soft Vote across 6 models ---
    # Average the threat probabilities
    all_threat_probs = [rf1_threat, lr1_threat, gb1_threat, rf2_threat, lr2_threat, gb2_threat]
    avg_threat = np.mean(all_threat_probs)
    
    # Determine Final Verdict
    final_verdict = "MALICIOUS / PHISHING" if avg_threat >= 0.5 else "SAFE / LEGITIMATE"
    final_confidence = avg_threat * 100 if avg_threat >= 0.5 else (1.0 - avg_threat) * 100

    # --- F. Display Results ---
    print("-" * 75)
    print("SET 1: GENERAL MALICIOUS MODELS (Trained on malicious_urls.csv)")
    print(f"  [1] Random Forest      : {rf1_lbl.upper():<12} (Confidence: {rf1_conf*100:.2f}%)")
    print(f"  [2] Logistic Regression: {lr1_lbl.upper():<12} (Confidence: {lr1_conf*100:.2f}%)")
    print(f"  [3] Gradient Boosting  : {gb1_lbl.upper():<12} (Confidence: {gb1_conf*100:.2f}%)")
    
    print("\nSET 2: PHISHING-SPECIFIC MODELS (Trained on dataset_phishing.csv)")
    print(f"  [4] Random Forest      : {rf2_lbl.upper():<12} (Confidence: {rf2_conf*100:.2f}%)")
    print(f"  [5] Logistic Regression: {lr2_lbl.upper():<12} (Confidence: {lr2_conf*100:.2f}%)")
    print(f"  [6] Gradient Boosting  : {gb2_lbl.upper():<12} (Confidence: {gb2_conf*100:.2f}%)")
    print("-" * 75)
    
    print("GLOBAL SOFT VOTING RESULTS (6-Model Ensemble):")
    print(f"COMBINED VERDICT : {final_verdict}")
    print(f"TOTAL CONFIDENCE : {final_confidence:.2f}%")
    
    if avg_threat >= 0.5:
        print("\n🚨 WARNING: The combined soft-vote indicates this URL is DANGEROUS.")
    else:
        print("\n✅ STATUS: The combined soft-vote indicates this URL is SAFE.")
        
    print("="*75 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_url(sys.argv[1])
    else:
        while True:
            url_to_test = input("Enter a URL to check (or type 'exit'): ")
            if url_to_test.lower() == 'exit': 
                break
            analyze_url(url_to_test)