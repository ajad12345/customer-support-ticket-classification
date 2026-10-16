# Customer Support Ticket Classification: Final Presentation Outline
**Team Batch 10 | Project 6: Customer Support Ticket Classification**  
*Format: 12 Slides with Slide Structure, Key Bullets, and Detailed Speaker Notes*

---

### Slide 1: Title & Team Introductions
- **Slide Title:** Intelligent Customer Support Ticket Classification & Automated Routing Engine
- **Subtitle:** An End-to-End Classical NLP Solution for Enterprise Support Triage
- **Batch & Course:** Team Batch 10, Project 6 | Natural Language Processing
- **Team Members:**
  - Student 1: Data Engineering & Text Preprocessing
  - Student 2: Model Benchmarking & Pipeline Architecture
  - Student 3: Hyperparameter Tuning, Evaluation & Error Analysis
  - Student 4: Streamlit Deployment & Production Triage UI
- **Speaker Notes:**
  > *"Good morning professors and fellow students. Today, Team Batch 10 is excited to present Project 6: Customer Support Ticket Classification. Our mission was to develop an autonomous, end-to-end NLP system that categorizes raw customer requests and automatically routes them to designated operational departments, while integrating a confidence gate for human-in-the-loop review. Best of all, we achieved this using purely classical machine learning—no external paid APIs or large language models."*

---

### Slide 2: Problem Statement & Industrial Motivation
- **Slide Title:** The Support Triage Bottleneck
- **Key Points:**
  - High inbound ticket volume across enterprise channels (email, web forms, mobile apps).
  - Manual routing is expensive, slow (average triage delay: 4–8 hours), and error-prone.
  - Naive keyword search fails on synonyms, misspellings, and negations.
  - Business Goal: Lower Mean Time to Resolution (MTTR) by routing requests instantaneously with calibrated confidence.
- **Speaker Notes:**
  > *"Every growing enterprise faces the support bottleneck. When thousands of tickets arrive daily, manual routing creates immense backlogs. A billing ticket sitting in the tech support queue wastes time and frustrates users. Keyword filters break easily—for instance, searching for 'receive' misroutes 'I did NOT receive my parcel'. Our goal is an intelligent, negation-aware NLP dispatcher that accurately routes tickets to the right team in milliseconds."*

---

### Slide 3: Target Support Taxonomy & Operational Routing
- **Slide Title:** Category Definitions & Department Mapping
- **Key Points (6 Core Categories):**
  1. `Billing` $\to$ **Finance & Accounts Operations** (unexpected charges, invoices, taxes)
  2. `Technical Support` $\to$ **Tier-2 Engineering & Tech Support** (crashes, 500 errors, bugs)
  3. `Account Access` $\to$ **Identity & Security Support** (password resets, 2FA/MFA, lockouts)
  4. `Order Issues` $\to$ **Logistics & Fulfillment Team** (shipping delays, lost parcels)
  5. `Refund/Returns` $\to$ **Returns & Reverse Logistics** (RMA requests, return shipping labels)
  6. `General Inquiry` $\to$ **Customer Care & Sales Advisory** (pricing, SLAs, business hours)
- **Speaker Notes:**
  > *"We defined a 6-category operational taxonomy mapping directly to typical enterprise teams. For instance, Billing routes to Finance, while Technical Support routes to Engineering. This clear separation allows our classifier to serve as a functional routing switch in production customer service platforms."*

---

### Slide 4: Dataset Engineering & Integrity
- **Slide Title:** Dataset Provenance, Diversity & Leakage Prevention
- **Key Points:**
  - 1,528 clean, unique records (250+ per class) curated with rich combinatorial templates.
  - Balanced class distribution: ~16.7% representation per category.
  - Natural diversity: short queries (2 words), long enterprise complaints (85+ words).
  - Linguistic features: realistic typos (25%), negations, and ambiguous boundary cases.
  - Strict deduplication performed **before** train/test split to prevent leakage.
- **Speaker Notes:**
  > *"To train our models without overfitting, we curated 1,528 unique customer tickets. We ensured balanced representation across all six classes to prevent majority-class bias. We explicitly injected diverse sentence lengths, realistic typos like 'subcription' or 'recieve', and cross-boundary cases. Crucially, exact and near-duplicates were eliminated before splitting to ensure complete data integrity."*

---

### Slide 5: NLP Preprocessing: The Negation-Aware Pipeline
- **Slide Title:** Linguistic Preprocessing Architecture
- **Key Points:**
  - **Lowercasing:** Normalizes casing variations.
  - **Entity Stripping (Regex):** Removes URLs, emails, transaction IDs (`#ORD-9921`), and dollar amounts (`$120`).
  - **Contraction Expansion:** Expands `can't` $\to$ `cannot` and `won't` $\to$ `will not`.
  - **Negation-Preserving Stopword Whitelist:** Whitelists `no`, `not`, `never`, `cannot`.
  - **WordNet Lemmatization:** Normalizes `crashes` $\to$ `crash` and `charged` $\to$ `charge`.
- **Speaker Notes:**
  > *"Preprocessing is where standard NLP projects often fail. Standard stopword removal blindly removes 'not' and 'no'. In customer service, turning 'I did NOT authorize this charge' into 'authorize charge' completely reverses the intent! Our pipeline expands contractions, whitelists negation tokens, strips entity noise like order IDs, and applies WordNet lemmatization to preserve dictionary-valid root words."*

---

### Slide 6: Feature Extraction & Zero Data Leakage
- **Slide Title:** TF-IDF Vectorization & Pipeline Encapsulation
- **Key Points:**
  - **Sublinear Term Frequency Scaling:** Replaces linear frequency with $1 + \log(\text{tf})$ to prevent high-frequency term dominance.
  - **Unigram + Bigram Context:** Captures phrases like `not receive`, `credit card`, `crash launch`.
  - **Pipeline Encapsulation:** Vectorizer is embedded inside a scikit-learn `Pipeline`.
  - **Zero Leakage Rule:** Vectorizer is fitted strictly on training folds; test folds are transformed only.
- **Speaker Notes:**
  > *"For text representation, we used TF-IDF with both unigrams and bigrams. Bigrams allow the model to recognize phrases like 'not receive' or 'server error'. More importantly, we enforced zero data leakage: by encapsulating TfidfVectorizer within a scikit-learn Pipeline, we ensure that inverse document frequencies and vocabulary are calculated strictly from training data, never leaking test set distribution."*

---

### Slide 7: Model Exploration & 5-Fold Stratified Cross-Validation
- **Slide Title:** Comparative Evaluation Across 3 Architectures
- **Key Points:**
  - Evaluated on 1,222 training samples using 5-Fold Stratified CV (`seed=42`).
  - **Multinomial Naive Bayes:** Mean CV Macro-F1 = 0.9943 (±0.0076)
  - **Logistic Regression (C=2.0):** Mean CV Macro-F1 = 0.9967 (±0.0066)
  - **Linear SVM (Calibrated):** Mean CV Macro-F1 = **1.0000 (±0.0000)**
  - Why Macro-F1? Equal weighting across all six classes prevents masking errors in smaller categories.
- **Speaker Notes:**
  > *"We benchmarked three diverse paradigms: generative Naive Bayes, discriminative Logistic Regression, and maximum-margin Linear SVM. We used 5-Fold Stratified Cross-Validation on the training partition. All three models performed exceptionally well, but Calibrated Linear SVM achieved a flawless 1.0000 Macro-F1 with zero variance across all folds. We prioritize Macro-F1 over accuracy because it treats every support department with equal weight."*

---

### Slide 8: Hyperparameter Optimization (GridSearchCV)
- **Slide Title:** Tuning the Champion Pipeline
- **Key Points:**
  - Systematic grid search over n-gram ranges `[(1, 1), (1, 2)]` and regularization `C: [0.5, 1.0, 2.0]`.
  - Best parameters: `ngram_range=(1, 1)`, `C=1.0`.
  - Platt Scaling (`CalibratedClassifierCV`): Transforms raw SVM decision margins into true posterior probabilities using cross-validated logistic regression.
- **Speaker Notes:**
  > *"To ensure optimal generalization, we ran GridSearchCV over our vectorizer and SVM hyperparameters. Additionally, because standard SVMs only output geometric margins rather than probabilities, we integrated Platt Scaling via CalibratedClassifierCV. This provides well-calibrated class probabilities, which is essential for our confidence gating mechanism."*

---

### Slide 9: Unseen Holdout Test Set Evaluation
- **Slide Title:** Final Benchmark on Holdout Data (306 Samples)
- **Key Points:**
  - **Holdout Test Accuracy:** 100.0% (306/306 correct)
  - **Holdout Macro Precision / Recall / F1:** 1.0000 / 1.0000 / 1.0000
  - Per-class F1-scores: 1.0000 across all 6 classes.
  - Zero misclassifications on clean domain test split.
  - All artifacts saved: `model.pkl`, `vectorizer.pkl`, `metrics.json`.
- **Speaker Notes:**
  > *"When evaluated on our 306 unseen holdout test samples, our champion model achieved 100% precision and recall across every single class. The confusion matrix confirms zero off-diagonal misclassifications on standard domain requests. All artifacts were persisted with Joblib and JSON for real-time serving in our web application."*

---

### Slide 10: Robustness Testing & Error Analysis
- **Slide Title:** Edge-Case Evaluation & Failure Mode Analysis
- **Key Points (16 Systematic Test Scenarios):**
  - **Input Validation:** Empty strings and short queries (`"hi"`) are cleanly intercepted.
  - **Gibberish Input:** `"asdfjk qwerty 12345"` yields 35.7% max confidence $\to$ **Blocked by threshold** $\to$ Sent to Manual Review.
  - **Negation Stress Tests:** *"I did NOT authorize this"* (97.0% Billing), *"NOT received"* (82.4% Order Issues).
  - **Mixed-Intent Ambiguity:** *"Double charged... want immediate refund"* routes to Refund/Returns (72.7%).
  - **Out-of-Domain Detection:** *"What is the weather tomorrow?"* yields low confidence (53.0%) $\to$ demonstrates safe boundary limitations.
- **Speaker Notes:**
  > *"We didn't stop at clean test data. We subjected our system to 16 stress-test scenarios. Empty inputs are caught by validation. Gibberish produces a flat 35.7% confidence and is blocked by our 50% confidence threshold, routing to human review. Negated tickets retain high confidence because our preprocessing preserves negations. And mixed-intent tickets are routed according to their primary action item."*

---

### Slide 11: Production Deployment: Streamlit Application
- **Slide Title:** Interactive Triage & Bulk Ingestion Interface
- **Key Points:**
  - **Cached Model Loading:** `@st.cache_resource` loads artifacts once; never retrains on submit.
  - **Live Triage Tab:** Pre-filled scenario buttons, real-time prediction, probability bar chart, and department routing badge.
  - **Confidence Gate Slider:** Adjustable threshold (0.20 to 0.95); auto-routes or redirects to human triage.
  - **Bulk CSV Tab:** Uploads batches of tickets, automatically classifies rows, displays triage summary, and exports enriched CSV.
  - **Analytics Tab:** Displays confusion matrix, model comparisons, and cross-validation tables.
- **Speaker Notes:**
  > *"We deployed our trained pipeline as an enterprise Streamlit web application. It features cached model persistence, meaning zero latency overhead on submit. Operators can test single tickets, adjust the confidence threshold slider, or upload bulk CSV files containing hundreds of tickets, exporting an enriched routing manifest with a single click."*

---

### Slide 12: Conclusion, Limitations & Future Enhancements
- **Slide Title:** Summary & Strategic Roadmap
- **Key Points:**
  - **Summary:** Delivered a complete, reproducible, high-performance classical NLP triage system.
  - **Limitations:** Monolingual English scope; single-label ground truth.
  - **Future Roadmap:**
    1. Multi-label / Hierarchical classification (Sub-intents).
    2. Customer sentiment and urgency scoring.
    3. Active learning feedback loop inside the Streamlit UI.
    4. Lightweight distilled Transformers (DistilBERT) exploration.
- **Speaker Notes:**
  > *"In conclusion, Project 6 demonstrates that properly engineered classical NLP with negation-aware preprocessing and calibrated SVMs can achieve outstanding real-world performance without the expense of LLMs. In future iterations, we plan to add multi-label triage, urgency scoring, and active learning feedback from support agents. Thank you, and we welcome your questions!"*
