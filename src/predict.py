"""
Prediction and Routing Inference Module
Team Batch 10, Project 6: Customer Support Ticket Classification

Loads persisted model pipeline and vectorizer artifacts to classify incoming
customer support requests in real-time or batch mode, with confidence scoring
and team routing.
"""

import os
import sys
import argparse
import joblib
import numpy as np

# Ensure project root in sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.preprocess import preprocess_text

# Department routing directory mapping categories to target operational teams
DEPARTMENT_ROUTING = {
    "Billing": "Finance & Accounts Operations",
    "Technical Support": "Tier-2 Technical Support & Engineering",
    "Account Access": "Identity, Access & Security Support",
    "Order Issues": "Logistics & Fulfillment Team",
    "Refund/Returns": "Returns & Reverse Logistics",
    "General Inquiry": "Customer Care & Sales Advisory"
}

CONFIDENCE_THRESHOLD = 0.50

class TicketClassifier:
    """Wrapper class for loaded model inference."""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "model.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Please run `python src/train.py` first to train and persist the model."
            )
            
        self.model = joblib.load(model_path)
        self.classes_ = list(self.model.classes_)
        
    def predict(self, text: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        """
        Classify a single ticket text.
        
        Args:
            text (str): Raw customer request text
            threshold (float): Minimum confidence for automatic routing
            
        Returns:
            dict containing predicted category, confidence, assigned department,
            flag indicating whether manual review is required, and class probabilities.
        """
        if not text or not text.strip() or len(text.strip()) < 3:
            return {
                "category": "Unassigned",
                "confidence": 0.0,
                "department": "Manual Triage Queue",
                "needs_manual_review": True,
                "clean_text": "",
                "status": "Rejected: Empty or insufficient input",
                "probabilities": {cls: 0.0 for cls in self.classes_}
            }
            
        cleaned = preprocess_text(text)
        if not cleaned:
            return {
                "category": "Unassigned",
                "confidence": 0.0,
                "department": "Manual Triage Queue",
                "needs_manual_review": True,
                "clean_text": "",
                "status": "Rejected: Text contained no meaningful lexical tokens after cleaning",
                "probabilities": {cls: 0.0 for cls in self.classes_}
            }
            
        # Pipeline handles TF-IDF vectorization and classification
        probabilities = self.model.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(probabilities))
        pred_class = self.classes_[pred_idx]
        confidence = float(probabilities[pred_idx])
        
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes_, probabilities)}
        
        needs_review = confidence < threshold
        assigned_dept = DEPARTMENT_ROUTING.get(pred_class, "General Operations")
        
        return {
            "category": pred_class,
            "confidence": round(confidence, 4),
            "department": assigned_dept if not needs_review else "Manual Review Required (Low Confidence)",
            "needs_manual_review": needs_review,
            "clean_text": cleaned,
            "status": "Success",
            "probabilities": prob_dict
        }

# Global singleton helper
_CLASSIFIER_INSTANCE = None

def get_classifier():
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = TicketClassifier()
    return _CLASSIFIER_INSTANCE

def classify_ticket(text: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Convenience function for quick single prediction."""
    return get_classifier().predict(text, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict category and routing for customer support tickets.")
    parser.add_argument("text", nargs="?", default=None, help="Customer ticket text to classify.")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold (default 0.50).")
    
    args = parser.parse_args()
    
    sample_text = args.text or "I was double charged $120 for invoice #99281 on my corporate Visa. Please issue an invoice credit."
    
    clf = get_classifier()
    res = clf.predict(sample_text, threshold=args.threshold)
    
    print("\n" + "=" * 60)
    print("CUSTOMER TICKET INFERENCE RESULT")
    print("=" * 60)
    print(f"Raw Input:       {sample_text}")
    print(f"Preprocessed:    {res['clean_text']}")
    print(f"Predicted Class: {res['category']}")
    print(f"Confidence:      {res['confidence'] * 100:.2f}%")
    print(f"Department:      {res['department']}")
    print(f"Manual Review?:  {'YES (Caution: Ambiguous/Low Confidence)' if res['needs_manual_review'] else 'NO (Eligible for Auto-Routing)'}")
    print("\nClass Probabilities:")
    for cat, p in sorted(res['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat:20s}: {p*100:6.2f}%")
    print("=" * 60)
