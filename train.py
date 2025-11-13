import pandas as pd
import numpy as np
import joblib
import re
import os  # <-- Import os for path joining
import kagglehub  # <-- Import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

# --- Custom Feature Extractors (No changes here) ---

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Calculates the length of the text."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        # Added str() to handle any potential non-string data
        return np.c_[np.array([len(str(text)) for text in x])] 

class PercentCapsExtractor(BaseEstimator, TransformerMixin):
    """Calculates the percentage of characters that are uppercase."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        def count_caps(text):
            text_str = str(text) # Ensure text is a string
            if len(text_str) == 0:
                return 0
            caps = sum(1 for char in text_str if char.isupper())
            return caps / len(text_str)
        
        return np.c_[np.array([count_caps(text) for text in x])]

class PercentDigitsExtractor(BaseEstimator, TransformerMixin):
    """Calculates the percentage of characters that are digits."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        def count_digits(text):
            text_str = str(text) # Ensure text is a string
            if len(text_str) == 0:
                return 0
            digits = sum(1 for char in text_str if char.isdigit())
            return digits / len(text_str)
        
        return np.c_[np.array([count_digits(text) for text in x])]

# --- Main Training Script ---

# 1) Download and load data
# --- THIS SECTION IS UPDATED TO AUTO-DOWNLOAD ---
print("Checking for dataset...\n")
try:
    # Download the dataset. It will be cached locally.
    dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    
    # Construct the full path to the specific CSV file
    csv_path = os.path.join(dataset_path, 'phishing_email.csv')
    
    print(f"Dataset found. Loading from: {csv_path}\n")
    
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Drop rows where text_combined or label is missing
    df.dropna(subset=['text_combined', 'label'], inplace=True)
    
    # Rename 'text_combined' to 'text' to fit the rest of the pipeline
    df.rename(columns={'text_combined': 'text'}, inplace=True)
    
    # Ensure the label column is integer (0 or 1)
    df['label'] = df['label'].astype(int)

    print(f"Data loaded successfully. Spam count: {df['label'].sum()}, Ham count: {len(df) - df['label'].sum()}")

except Exception as e:
    print(f"Error downloading or loading data: {e}")
    print("\nPlease ensure you have run 'pip install kagglehub'")
    print("and that your Kaggle API token (kaggle.json) is in the correct location (e.g., C:\\Users\\YourName\\.kaggle\\).")
    exit()
# --- END OF UPDATED SECTION ---


# 2) Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Build the preprocessing and feature extraction pipeline
text_features = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2) 
)

feature_transformer = FeatureUnion([
    ('tfidf_vectorizer', text_features),
    ('text_length', TextLengthExtractor()),
    ('percent_caps', PercentCapsExtractor()),
    ('percent_digits', PercentDigitsExtractor())
])

# 4) Create the full pipeline
pipeline = Pipeline([
    ('features', feature_transformer),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced'))
])

# 5) Set up GridSearchCV for hyperparameter tuning
parameters = {
    'features__tfidf_vectorizer__max_features': (None, 10000), 
    'features__tfidf_vectorizer__min_df': (1, 5),                  
    'clf__C': (0.1, 1, 10),                                        
}

grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1, scoring='f1')
print("\nStarting model training and grid search on the full dataset...")
print("This will take significantly longer than before.")
grid_search.fit(X_train, y_train)

# 6) Evaluate the best model
print("\n--- Model Evaluation ---")
print(f"Best parameters found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_pred, target_names=['Legit (Ham)', 'Spam']))

# 7) Save the *entire* pipeline
joblib.dump(best_model, 'phishing_model.joblib')
print("\nSuccessfully trained and saved the new, more powerful model to 'phishing_model.joblib'")