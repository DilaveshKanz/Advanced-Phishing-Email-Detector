import joblib
import sys
import argparse
import numpy as np  # <-- Added this import
from sklearn.base import BaseEstimator, TransformerMixin  # <-- Added this import

# --- ADD THESE CUSTOM CLASSES ---
# These definitions must be present for joblib to load the pipeline

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Calculates the length of the text."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        # np.c_ converts the list of lengths to a 2D NumPy array
        return np.c_[np.array([len(text) for text in x])]

class PercentCapsExtractor(BaseEstimator, TransformerMixin):
    """Calculates the percentage of characters that are uppercase."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        def count_caps(text):
            if len(text) == 0:
                return 0
            caps = sum(1 for char in text if char.isupper())
            return caps / len(text)
        
        return np.c_[np.array([count_caps(text) for text in x])]

class PercentDigitsExtractor(BaseEstimator, TransformerMixin):
    """Calculates the percentage of characters that are digits."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        def count_digits(text):
            if len(text) == 0:
                return 0
            digits = sum(1 for char in text if char.isdigit())
            return digits / len(text)
        
        return np.c_[np.array([count_digits(text) for text in x])]

# --- END OF ADDED CLASSES ---


def load_model(model_path='phishing_model.joblib'):
    """
    Loads the entire scikit-learn pipeline from disk.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'training.py' first to create the model file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_email(text, model, phishing_thresh=0.5):
    """
    Predicts if a single email text is phishing or legit using the loaded pipeline.
    
    Note: The original heuristic for short messages is removed, as the new
    'TextLengthExtractor' feature in the model handles this automatically.
    """
    
    # The model pipeline expects a list or array of texts
    text_to_predict = [text]
    
    # Get probabilities: [prob_legit, prob_phishing]
    # The pipeline handles all vectorizing and feature extraction internally.
    probs = model.predict_proba(text_to_predict)[0]
    
    # Get the probability of class 1 (phishing)
    p_phishing = probs[1] 

    # --- THIS IS THE CORRECTED LOGIC ---
    if p_phishing >= phishing_thresh:
        # If phishing probability is HIGH, it's PHISHING
        return "PHISHING", p_phishing
    else:
        # Otherwise, it's LEGIT
        return "LEGIT", (1 - p_phishing) # Confidence in it being legit

def main():
    parser = argparse.ArgumentParser(description='Spam/Phishing Email Detector CLI')
    parser.add_argument('email', nargs='?', type=str, help='Email text to analyze (enclose in quotes)')
    parser.add_argument('--file', type=str, help='Path to text file containing email')
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.45, 
        help='Probability threshold to classify as phishing (0.0 to 1.0). Default is 0.5'
    )
    args = parser.parse_args()

    # Load the entire model pipeline
    model = load_model()

    text = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif args.email:
        text = args.email
    else:
        print("\nPaste the email content below. Press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) to finish:\n")
        try:
            lines = sys.stdin.readlines()
            text = "".join(lines)
        except EOFError:
            pass

    if not text.strip():
        print("No input text provided. Exiting.")
        sys.exit(1)

    result, confidence = predict_email(text, model, args.threshold)

    print(f"\n{'='*30}")
    if result == 'PHISHING':
        print(f"Result:     🚨 PHISHING 🚨")
    else:
        print(f"Result:     ✅ LEGIT")
        
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"{'='*30}\n")


if __name__ == '__main__':
    main()