<h1 align="center">Advanced Phishing Email Detector</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Kaggle-Dataset-lightblue?style=for-the-badge&logo=kaggle">
</p>

<p align="center">
A Python-based command-line tool that detects whether an email is <b>Phishing</b> or <b>Legitimate</b> using a machine learning model trained on real-world data.
</p>

---

## 📘 Overview

This tool uses a **scikit-learn ML pipeline** that combines **TF-IDF text analysis** and **statistical metadata** to detect phishing emails.  
It is trained on the [Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset), which contains over **82,000 email samples**.

---

## 🚀 Features

✅ **Large, Real Dataset** — 82K+ labeled emails for robust model performance  
🧠 **Hybrid Feature Engineering** — Combines:
- **Text Features:** Word/phrase frequency via `TfidfVectorizer` (1-2 grams)
- **Statistical Features:** Email length, % of capitals, % of digits  
⚙️ **Optimized Model:** Uses `GridSearchCV` to fine-tune `LogisticRegression`  
💻 **Command-Line Interface:** Analyze text, files, or pasted emails directly  
📈 **Custom Sensitivity:** Adjust phishing detection threshold

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies

You’ll need **Python 3.8+** and the following libraries:

```bash
pip install pandas scikit-learn numpy joblib kagglehub
```

---

### 2️⃣ Set Up Kaggle API (Required for Training)

The script **`train.py`** downloads the dataset automatically via the Kaggle API.  

1. Log in to [Kaggle](https://www.kaggle.com/).  
2. Go to **Account → API → Create New API Token**.  
3. This downloads a `kaggle.json` file.  
4. Move it to one of these paths:
   - **Windows:** `C:\Users\<Your-Username>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`

---

## 🧪 How to Use

Run the **`app.py`** script in one of the following modes:

### ▶️ 1. Quick String Check
```bash
python app.py "Congratulations! You've won a $500 Amazon gift card."
```

### 📄 2. Analyze from a File
```bash
python app.py --file my_email.txt
```

### 📋 3. Paste Full Email
```bash
python app.py
```
Paste your content, then press:
- **Ctrl + Z** → Enter (Windows)
- **Ctrl + D** (macOS/Linux)

---

## 🎚️ Adjust Sensitivity (Threshold)

By default, the detector flags an email as *Phishing* if its confidence ≥ **0.45**.  
You can adjust this with `--threshold`:

**Be more strict (less likely phishing):**
```bash
python app.py --threshold 0.8 "This looks safe"
```

**Be more sensitive (more likely phishing):**
```bash
python app.py --threshold 0.3 "This might be phishing"
```

---

## 🧾 Example Outputs

**Phishing Example:**
```
==============================
Result:     🚨 PHISHING 🚨
Confidence: 92.45%
==============================
```

**Legit Example:**
```
==============================
Result:     ✅ LEGIT
Confidence: 99.81%
==============================
```

---

## 🧠 Re-Train the Model

To train from scratch, simply run:

```bash
python train.py
```

This will:
1. Download the **phishing_email.csv** dataset via Kaggle.  
2. Run **GridSearchCV** to optimize the model.  
3. Print performance metrics.  
4. Save the trained model as **`phishing_model.joblib`**.

---

