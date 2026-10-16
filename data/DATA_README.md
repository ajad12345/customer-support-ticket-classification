# Customer Support Tickets Dataset Documentation

## 1. Overview and Provenance
- **Dataset Name**: Customer Support Ticket Intent Dataset (Team Batch 10, Project 6)
- **Source**: Self-created synthetic benchmark designed for supervised text classification and automated ticket triage in enterprise customer operations.
- **File**: `data/tickets.csv`
- **Total Records**: 1,535 clean, deduplicated examples
- **Total Features**: 3 columns (`ticket_id`, `text`, `category`)
- **Random Seed**: 42 (used across generation, splitting, and modeling)

## 2. Schema Definition
| Column Name | Data Type | Description | Sample Value |
|---|---|---|---|
| `ticket_id` | String | Unique ticket identifier formatted as `TCK-XXXXX` | `TCK-10042` |
| `text` | String | Customer message body containing issue description, error code, or inquiry | *"I was charged $120 on my card, but my subscription was cancelled."* |
| `category` | String | Target ground truth classification label (6 categories) | `Billing` |

## 3. Label Categories & Descriptions
1. **Billing**:
   - *Scope*: Unexpected charges, recurring subscription deductions, invoice/VAT requests, currency conversion, corporate payment failure, promo discount issues.
   - *Example*: *"Why was I billed twice for invoice #49201? Please check transaction #TXN819201."*
2. **Technical Support**:
   - *Scope*: Application crashes, HTTP 500 errors, rate-limiting, database timeouts, mobile freezing, webhook timeouts, memory leaks.
   - *Example*: *"The application crashes every time I click the 'Export to PDF' button on macOS."*
3. **Account Access**:
   - *Scope*: Password resets, 2FA/MFA token errors, account lockouts, corporate email transfers, SSO SAML failures, session expiration loops.
   - *Example*: *"I am locked out of my account after entering my password incorrectly three times."*
4. **Order Issues**:
   - *Scope*: Package tracking delays, damaged shipments, missing items, address updates, incorrect item delivery, courier carrier disputes.
   - *Example*: *"My package for order #ORD-49201 has been stuck in 'Label Created' status for over seven days."*
5. **Refund/Returns**:
   - *Scope*: Return authorization (RMA), prepaid shipping labels, store credit vs bank refund disputes, restocking fees, warranty replacements.
   - *Example*: *"I returned the defective laptop two weeks ago (tracking #TRK9920184); when will my refund be processed?"*
6. **General Inquiry**:
   - *Scope*: Support working hours, SLA commitments, SOC 2 / GDPR compliance certifications, API documentation, demo scheduling, academic pricing.
   - *Example*: *"Does your platform comply with GDPR, HIPAA, and SOC 2 Type II data privacy standards?"*

## 4. Class Distribution
The dataset is well-balanced across all 6 classes:
- `Billing`: 259 samples (16.9%)
- `Refund/Returns`: 258 samples (16.8%)
- `Order Issues`: 258 samples (16.8%)
- `Technical Support`: 257 samples (16.7%)
- `Account Access`: 253 samples (16.5%)
- `General Inquiry`: 250 samples (16.3%)
- **Total**: 1,535 samples

## 5. Dataset Diversity & Design Considerations
To ensure high real-world applicability without synthetic degradation:
- **Length Diversity**: Samples range from ultra-short queries (2-4 words, e.g., *"cannot login"*) to complex multi-sentence enterprise complaints (>80 words).
- **Negation Handling**: Natural negations are prominently featured (*"I did NOT authorize this payment"*, *"Tracking has not updated"*, *"Button does not respond"*).
- **Noise & Misspellings**: Approximately 20-30% of records incorporate realistic typos (*"recieve"*, *"acount"*, *"subcription"*, *"dosent"*).
- **Boundary Ambiguity**: Intentional overlapping terminology exists (e.g., requesting a refund for an incorrect billing deduction vs returning a physical product) to simulate genuine customer service edge cases.

## 6. Deduplication and Leakage Prevention
- Prior to train-test splitting, all records underwent lowercase alphanumeric normalization and exact/near-duplicate pruning.
- A 100% disjoint Stratified 80/20 train-test split is enforced in the model training pipeline (`random_state=42`), ensuring zero information leakage between the training corpus and evaluation test set.

## 7. Real-World Limitations
1. **Synthetic Nature**: Generated using curated domain seed templates; real customer tickets may contain slang, emojis, audio transcriptions, or OCR scan artifacts not captured here.
2. **Monolingual Focus**: Strictly English-language tickets. Multilingual inputs require translation or cross-lingual embeddings.
3. **Single-Label Assumption**: Each ticket is mapped to exactly one primary category. In enterprise production, some tickets involve multi-intent requests (e.g., both technical bug and billing refund request).
