# Test Cases & Empirical Model Evaluation

This document outlines the systematic test suite for the **Customer Support Ticket Classifier** (Team Batch 10, Project 6). It includes 16 distinct test scenarios covering standard domain inputs, edge cases, input validation, complex negations, mixed-intent requests, and out-of-scope queries.

All results, confidence scores, and department routing assignments are recorded from actual execution against the trained champion model pipeline.

---

## 1. Test Summary Table

| Test Case | Category Type | Input Text Snippet | Predicted Category | Confidence | Auto-Route Flag | Operational Routing Team | Behavioral Observation |
|---|---|---|---|---|---|---|---|
| **TC-01** | Normal Billing | *"I noticed an unexpected charge of $89 on my credit card statement this morning."* | **Billing** | 97.7% | Passed (Auto) | Finance & Accounts Operations | High-confidence match on financial deduction vocabulary. |
| **TC-02** | Normal Tech Support | *"The mobile app crashes every time I tap the export analytics button on Android."* | **Technical Support** | 98.5% | Passed (Auto) | Tier-2 Technical Support & Engineering | High-confidence match on crash, app, and platform terms. |
| **TC-03** | Normal Account Access | *"I forgot my password and the reset email is not arriving in my inbox."* | **Account Access** | 99.3% | Passed (Auto) | Identity, Access & Security Support | Clear intent match on credential recovery. |
| **TC-04** | Normal Order Issues | *"My package for order #ORD-48291 has been stuck at the regional distribution center for 6 days."* | **Order Issues** | 98.2% | Passed (Auto) | Logistics & Fulfillment Team | Accurate routing to logistics despite order tracking ID stripping. |
| **TC-05** | Normal Refund/Returns | *"I would like to return the shoes I ordered and need a prepaid return shipping label."* | **Refund/Returns** | 96.9% | Passed (Auto) | Returns & Reverse Logistics | Return RMA and label intent detected. |
| **TC-06** | Normal General Inquiry | *"What are your customer support working hours and telephone numbers?"* | **General Inquiry** | 95.5% | Passed (Auto) | Customer Care & Sales Advisory | Routed to general customer care. |
| **TC-07** | Empty Input | `""` (whitespace only) | **Unassigned** | 0.0% | Blocked | Manual Triage Queue | Intercepted by front-end validation check; no model call triggered. |
| **TC-08** | Very Short Input | *"hi"* | **Unassigned** | 0.0% | Blocked | Manual Triage Queue | Minimum character threshold (<3 chars) enforced; prevents garbage predictions. |
| **TC-09** | Gibberish Input | *"asdfjk qwerty zxcvbnm 12345"* | **Technical Support** | 35.7% | Blocked (<50%) | Manual Review Required | Low-confidence fallback triggered; blocked from auto-routing. |
| **TC-10** | Negated Billing | *"I did NOT authorize this renewal fee and I was promised I would not be charged."* | **Billing** | 97.0% | Passed (Auto) | Finance & Accounts Operations | Negation-preserving stopwords retain `'not authorize'` and `'not charge'`. |
| **TC-11** | Negated Order Issue | *"The courier says it was delivered, but I have NOT received any package at my door."* | **Order Issues** | 82.4% | Passed (Auto) | Logistics & Fulfillment Team | Negation preserved (`not receive`); correctly prevents confusion with delivery success. |
| **TC-12** | Negated Tech Support | *"The save button is not working and data does not sync across my devices."* | **Technical Support** | 76.2% | Passed (Auto) | Tier-2 Technical Support & Engineering | Multiple negative predicates (`not work`, `not sync`) correctly processed. |
| **TC-13** | Mixed-Category (Billing + Refund) | *"I was double charged on my card and I want an immediate refund back to my bank account."* | **Refund/Returns** | 72.7% | Passed (Auto) | Returns & Reverse Logistics | Overlapping intent between Billing and Refund; dominant request keyword (*refund*) resolved. |
| **TC-14** | Mixed-Category (Tech + Account) | *"The sign in page gives an HTTP 500 internal server error when I submit my password."* | **Account Access** | 79.6% | Passed (Auto) | Identity, Access & Security Support | Authentication workflow prioritized over generic error 500 keyword. |
| **TC-15** | Very Long Complaint | *"Dear Customer Service Director, Our enterprise organization has been using your software for over two years across 150 team members. Last Tuesday during our end-of-quarter financial closing, our administrators were abruptly locked out with invalid token errors, causing a massive delay. Furthermore, our monthly statement reflected an unexpected bandwidth surcharge of $450 that our contracted SLA explicitly waived. We demand an urgent call with technical management and a credit memo."* | **Billing** | 73.8% | Passed (Auto) | Finance & Accounts Operations | Long text with multi-intent elements accurately classified based on financial surcharge density. |
| **TC-16** | Out-of-Scope (OOD) | *"What is the weather forecast for New York City tomorrow afternoon?"* | **General Inquiry** | 53.0% | Passed (Borderline) | Customer Care & Sales Advisory | Demonstrates out-of-domain limitation: classified with reduced confidence (53.0%) as General Inquiry. |

---

## 2. Key Insights and Edge-Case Analysis

### A. Negation Handling Efficacy
Standard NLP pipelines that utilize off-the-shelf stopword lists frequently strip words such as `"not"`, `"no"`, and `"never"`. In our tests (TC-10, TC-11, TC-12), tickets with prominent negations maintained robust confidence (76.2% – 97.0%). For example, *"I have NOT received any package"* maps cleanly to **Order Issues**, whereas blindly dropping the negation would alter the semantic representation to `"receive package"`, risking misclassification as an inquiry or positive delivery status.

### B. Fallback Behavior and Input Validation
1. **Empty & Ultra-Short Inputs**: Tests TC-07 and TC-08 confirm that inputs with zero semantic tokens are intercepted immediately before vectorization, returning a clean user-facing error and routing to the manual triage queue.
2. **Gibberish / OOD Robustness**: TC-09 demonstrates that unstructured character sequences produce flat probability distributions (highest class probability = 35.7%). Because this falls below the system's **50.0% confidence threshold**, automated routing is inhibited and the ticket is escalated for human inspection.

### C. Multi-Intent & Lexical Ambiguity
In enterprise operations, customer communications are rarely single-topic. TC-13 and TC-14 illustrate that when cross-category terminology is present (e.g. *double charged* vs *immediate refund*), TF-IDF unigrams and bigrams balance feature weights effectively to route to the operational queue equipped to resolve the primary action item.
