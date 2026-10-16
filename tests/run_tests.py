"""
Test Suite Runner for Customer Support Ticket Classifier
Runs 16 diverse test cases through the real inference pipeline and records
actual outputs, confidence levels, and behavior observations.
"""

import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.predict import get_classifier

def run_tests():
    clf = get_classifier()
    
    test_cases = [
        ("Normal Billing", "I noticed an unexpected charge of $89 on my credit card statement this morning.", "Billing"),
        ("Normal Tech Support", "The mobile app crashes every time I tap the export analytics button on Android.", "Technical Support"),
        ("Normal Account Access", "I forgot my password and the reset email is not arriving in my inbox.", "Account Access"),
        ("Normal Order Issues", "My package for order #ORD-48291 has been stuck at the regional distribution center for 6 days.", "Order Issues"),
        ("Normal Refund/Returns", "I would like to return the shoes I ordered and need a prepaid return shipping label.", "Refund/Returns"),
        ("Normal General Inquiry", "What are your customer support working hours and telephone numbers?", "General Inquiry"),
        ("Empty Input", "   ", "Rejected"),
        ("Very Short Input", "hi", "Rejected"),
        ("Gibberish Input", "asdfjk qwerty zxcvbnm 12345", "Rejected / Uncertain"),
        ("Negated Billing", "I did NOT authorize this renewal fee and I was promised I would not be charged.", "Billing"),
        ("Negated Order Issue", "The courier says it was delivered, but I have NOT received any package at my door.", "Order Issues"),
        ("Negated Tech Support", "The save button is not working and data does not sync across my devices.", "Technical Support"),
        ("Mixed Billing & Refund", "I was double charged on my card and I want an immediate refund back to my bank account.", "Billing / Refund"),
        ("Mixed Tech & Account", "The sign in page gives an HTTP 500 internal server error when I submit my password.", "Technical Support / Account Access"),
        ("Very Long Complaint", "Dear Customer Service Director, Our enterprise organization has been using your software for over two years across 150 team members. Last Tuesday during our end-of-quarter financial closing, our administrators were abruptly locked out with invalid token errors, causing a massive delay. Furthermore, our monthly statement reflected an unexpected bandwidth surcharge of $450 that our contracted SLA explicitly waived. We demand an urgent call with technical management and a credit memo.", "Enterprise Escalation"),
        ("Out-of-Scope Input", "What is the weather forecast for New York City tomorrow afternoon?", "Out of Scope")
    ]
    
    results = []
    
    print("=" * 100)
    print("SYSTEMATIC TEST SUITE EXECUTION REPORT")
    print("=" * 100)
    
    for case_type, text, expected in test_cases:
        res = clf.predict(text, threshold=0.50)
        
        # Determine observation
        if res["status"].startswith("Rejected"):
            obs = f"Input validation rejected ({res['status']})"
        elif res["needs_manual_review"]:
            obs = f"Triggered manual review (Confidence {res['confidence']*100:.1f}% < 50%)"
        elif "Mixed" in case_type:
            obs = f"Resolves overlapping intent to '{res['category']}' ({res['confidence']*100:.1f}%)"
        elif case_type == "Out-of-Scope Input":
            if res["needs_manual_review"]:
                obs = f"Low confidence ({res['confidence']*100:.1f}%), correctly flagged for manual review"
            else:
                obs = f"Classified as '{res['category']}' ({res['confidence']*100:.1f}%), highlights boundary limitation for OOD inputs"
        else:
            obs = f"Successfully classified as '{res['category']}' ({res['confidence']*100:.1f}%)"
            
        results.append({
            "test_type": case_type,
            "input_text": text,
            "expected": expected,
            "predicted": res["category"],
            "confidence": res["confidence"],
            "department": res["department"],
            "needs_review": res["needs_manual_review"],
            "observation": obs,
            "clean_text": res["clean_text"]
        })
        
        print(f"[{case_type.upper()}]")
        print(f"  Input:       {text[:75]}{'...' if len(text) > 75 else ''}")
        print(f"  Predicted:   {res['category']} (Conf: {res['confidence']*100:.1f}%)")
        print(f"  Department:  {res['department']}")
        print(f"  Review Flag: {res['needs_manual_review']}")
        print(f"  Observation: {obs}\n")
        
    return results

if __name__ == "__main__":
    run_tests()
