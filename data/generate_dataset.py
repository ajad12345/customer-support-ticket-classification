"""
Dataset Generation Script for Customer Support Ticket Classification
Team Batch 10, Project 6: Customer Support Ticket Classification

Generates a realistic, diverse, synthetic dataset of support tickets across 6 categories:
1. Billing
2. Technical Support
3. Account Access
4. Order Issues
5. Refund/Returns
6. General Inquiry

Each category includes:
- Short, medium, and long tickets
- Different emotional tones (polite, urgent, frustrated, formal)
- Realistic typos and informal phrasing
- Complex negations ('I did NOT authorize...', 'Cannot login...', 'Package was not delivered')
- Ambiguous / boundary cases (e.g. billing vs refund)
"""

import os
import random
import re
import pandas as pd

# Set fixed random seed for full reproducibility
SEED = 42
random.seed(SEED)

CATEGORIES = [
    "Billing",
    "Technical Support",
    "Account Access",
    "Order Issues",
    "Refund/Returns",
    "General Inquiry"
]

# Vocabulary & phrasings per category
BILLING_SEEDS = [
    # Subscriptions & unexpected charges
    "I was charged ${amt} on my credit card, but my subscription was supposed to be cancelled last month.",
    "Why was I billed twice for invoice #{inv}? Please check transaction #{tx}.",
    "I noticed an unexpected charge of ${amt} on my bank statement from your company.",
    "My monthly billing cycle renewed automatically even though I turned off auto-renew in settings.",
    "I upgraded my plan yesterday but got charged full price instead of the prorated discount.",
    "Can you explain the extra fee of ${amt} listed on my latest billing statement #{inv}?",
    "I am seeing recurring deductions of ${amt} every week instead of monthly.",
    "There is an unauthorized charge on my account ending in {card}. I did not authorize this payment.",
    "Please send an itemized VAT invoice for order #{inv} to my finance department.",
    "My company requires a tax invoice with our corporate GSTIN/EIN number for annual tax deduction.",
    # Payment method issues
    "My credit card expired and when I try to update payment methods it declines without an error message.",
    "I keep getting payment failed notifications even though my bank confirms the funds are available.",
    "Can I pay my annual enterprise subscription via ACH wire transfer or corporate cheque?",
    "Why was my discount promo code '{code}' not applied at checkout? I was charged the regular price.",
    "I was told this promotional rate of ${amt}/mo would last 12 months, but it jumped to ${amt2} this month.",
    "You charged me for two active seats, but my colleague left the company three weeks ago.",
    "Please update our default billing currency to EUR instead of USD on company account #{acct}.",
    "I am disputing the charge on transaction ref #{tx}. This does not match our contracted tier.",
    "Failed payment warning: System says my debit card ending in {card} is invalid, but it works elsewhere.",
    "Can you split my invoice #{inv} across two corporate payment cards?",
    # Short variants
    "overcharged on invoice",
    "wrong subscription price billed",
    "unknown charge of ${amt} on statement",
    "need tax invoice for accounting",
    "promo code did not apply to bill",
    "payment method declined repeatedly",
    "double charged on card ending {card}",
    "why did my monthly subscription increase?",
    # Cross-boundary ambiguous cases (Billing vs Refund vs Account)
    "I was charged twice after returning my package, please check invoice #{inv} and reverse the duplicate deduction.",
    "My credit card was billed for an order that was cancelled before delivery; please reverse payment on card {card}.",
    "The subscription fee was charged to my card while my account was locked and I could not log in.",
    "There is an unexpected fee on my bill for a replacement shipment that was supposed to be completely free.",
    "We were charged for 5 licenses but our software crashed and could not be used; we need an invoice credit.",
    "Can you adjust our invoice #{inv}? We were double charged when returning item #{order}.",
    # Long & detailed
    "Hello Support Team, I am writing to formally dispute an overcharge on our latest monthly statement. We were billed ${amt} on invoice #{inv} dated last Tuesday. According to our agreement, our base monthly tier is strictly fixed, yet there is a line item for additional bandwidth that we did not utilize. Could someone from accounts review the ledger and issue a revised bill?",
    "Dear Accounts, We noticed two separate deductions of ${amt} on our corporate Mastercard statement on the 12th. Both refer to reference #{tx}. It appears the payment gateway submitted the transaction twice during server checkout. Please rectify this immediately and send confirmation."
]

TECH_SUPPORT_SEEDS = [
    # Crashes, bugs, performance
    "The application crashes every time I click the 'Export to PDF' button on macOS.",
    "I am receiving an HTTP 500 Internal Server Error whenever I navigate to the analytics dashboard.",
    "The desktop client freezes completely when importing a CSV file containing more than 5,000 rows.",
    "API integration failure: Getting '429 Rate Limit Exceeded' despite being well within our quota.",
    "The mobile app on Android 14 closes unexpectedly immediately after the splash screen appears.",
    "Our webhooks are failing with connection timeout errors since your maintenance window yesterday.",
    "The data sync between our iOS app and cloud database has stopped working since 9:00 AM.",
    "Page not rendering correctly: The main navigation sidebar overlaps content on Safari browser.",
    "Getting 'Database connection error: timeout after 30000ms' while querying historical logs.",
    "The search filter does not return any results when searching by date range, even though records exist.",
    "Memory leak issue: The application consumes over 6GB of RAM after running for two hours in the background.",
    "SSL certificate validation failed error when connecting to your API endpoint https://api.service.io/v2.",
    "The real-time notification socket keeps disconnecting every 30 seconds with error code 1006.",
    "My team cannot save edits to shared project boards. It says 'Conflict: document locked by another user'.",
    "Audio input is not detected during screen recording sessions on Windows 11.",
    "The plugin fails to load inside VS Code with error: 'Extension host terminated unexpectedly'.",
    "Graph visualization fails to display axes when rendering datasets with negative coordinates.",
    "Bulk upload tool throws unhandled TypeError: cannot read property 'map' of undefined.",
    "Search index appears corrupted; newly created tickets are not searchable by any keyword.",
    "Batch processing job #{job} has been stuck in 'In Progress' status for over 18 hours.",
    # Short variants
    "app keeps crashing on launch",
    "500 internal server error on dashboard",
    "export button not responding",
    "webhook payload delivery timeout",
    "cloud sync stopped working",
    "blank white screen after login",
    "API endpoint returning 502 bad gateway",
    "mobile app freezing on startup",
    # Cross-boundary ambiguous cases (Tech vs Account vs Billing)
    "The login page throws an unhandled exception and 500 server error when entering password.",
    "The billing checkout page crashes with an uncaught runtime error when processing payment.",
    "API endpoint for shipping and order tracking fails with an internal database timeout.",
    "Our user cannot log in because the client application freezes on the authentication screen.",
    # Long & detailed
    "Hi Technical Support, We have encountered a reproducible bug in the production environment. When our automated pipeline submits batch payload #{job}, the background worker throws an OutOfMemory exception and terminates without rolling back the transaction. We have attached the server stack trace and reproduction steps. This is halting our daily reporting.",
    "Urgent technical assistance needed: Since upgrading our desktop application to version 4.2.1, the CPU utilization spikes to 100% on all cores whenever background indexing starts. Reverting to 4.2.0 temporarily resolves the problem. Please inspect the attached diagnostic dump."
]

ACCOUNT_ACCESS_SEEDS = [
    # Login, credentials, 2FA
    "I am locked out of my account after entering my password incorrectly three times.",
    "The two-factor authentication (2FA) SMS code is not arriving on my mobile phone.",
    "I did not receive the password reset link at my registered email address {email}.",
    "My authenticator app OTP is rejected as invalid even though the device time is synchronized.",
    "I need to change the primary email address on my corporate profile from {email} to {email2}.",
    "Single Sign-On (SSO) authentication fails with error: 'SAML assertion response signature invalid'.",
    "Our former project manager has left the company; we urgently need to transfer workspace ownership.",
    "I forgot my security questions and cannot bypass the account recovery checkpoint.",
    "My account was temporarily suspended for suspicious activity, but I was just traveling abroad.",
    "Getting error: 'Your session has expired, please log in again', but logging in loops back to the same page.",
    "How can I revoke active sessions across all devices? I suspect my account might be compromised.",
    "I cannot verify my identity because the upload photo verification page gives an upload error.",
    "Our team admin cannot invite new team members; the button says 'User invitation quota exceeded'.",
    "The magic link sent to my inbox has already expired within 2 minutes of being requested.",
    "I lost my phone with Google Authenticator and did not save my backup recovery codes.",
    "Access denied: 'You do not have permission to access this organization's workspace'.",
    "Can you please unlock my profile #{user}? I have urgent client meetings this morning.",
    "The CAPTCHA image will not load on the login page, preventing me from submitting my password.",
    "My user profile is stuck in 'Pending Verification' status for more than 4 days.",
    "I need to remove phone number verification from my profile and switch to security keys.",
    # Short variants
    "cannot log in to my account",
    "password reset email not arriving",
    "2FA code SMS not received",
    "account locked after failed attempts",
    "SSO login failure with Okta",
    "transfer admin ownership to another user",
    "session expired redirect loop",
    "lost authenticator recovery keys",
    # Cross-boundary ambiguous cases (Account vs Tech vs Billing)
    "I cannot access my account to update our corporate billing credit card.",
    "Authentication error 401: cannot log in to track my active delivery order.",
    "I am locked out of our workspace where our technical API keys and logs are stored.",
    "The password reset screen gives a server timeout error when I submit my email.",
    # Long & detailed
    "Hello Team, I have been locked out of my developer account for the past 6 hours. I requested a password reset token to {email}, but nothing has arrived in my primary inbox or spam folder. Our company has strict firewall rules, so please check if your mail delivery IP has been throttled or provide a direct manual reset link.",
    "Dear Help Desk, Our enterprise team is unable to authenticate via our identity provider this morning. When users attempt to sign in via Google Workspace SSO, they get error code AUTH_403 'SAML assertion expired'. We need immediate escalation to resolve this company-wide access lockout."
]

ORDER_ISSUES_SEEDS = [
    # Delivery, shipping, packaging
    "My package for order #{order} has been stuck in 'Label Created' status for over seven days.",
    "The tracking information says my parcel was delivered to my porch, but nothing is here.",
    "I received the completely wrong item in package #{order}; I ordered a monitor but got cables.",
    "One of the items is missing from our bulk shipment #{order}. The invoice shows 5 units, box had 4.",
    "The shipping address on order #{order} has an incorrect apartment number; please correct it before shipment.",
    "My delivery has been delayed three times without any update from the courier carrier.",
    "The delivery driver marked 'Customer unavailable' but I was at home waiting all afternoon.",
    "Package arrived with severe physical damage to the outer carton and the inner contents are broken.",
    "Can I upgrade my shipping method on order #{order} from standard ground to next-day express?",
    "The courier attempted delivery at 10 PM after business hours and left no delivery notice.",
    "Tracking number #{track} returns an error stating 'Carrier number not found in postal system'.",
    "I need to cancel order #{order} immediately as it has not been fulfilled or dispatched yet.",
    "The estimated delivery date has passed by five days and tracking shows 'Held at customs facility'.",
    "Why was my order split into three separate deliveries with different tracking numbers?",
    "Can I reroute package #{order} to a local pickup locker instead of my residential address?",
    "The package was delivered to the wrong building down the street according to the GPS delivery photo.",
    "Order #{order} shows 'Fulfillment processing' for two weeks without dispatch confirmation.",
    "I requested signature-required delivery, but the package was dumped on the street curbside.",
    "Can you confirm whether order #{order} has actually left your regional distribution warehouse?",
    "The fragile ceramic items arrived shattered due to insufficient bubble wrap inside the parcel.",
    # Short variants
    "package tracking stuck in transit",
    "marked delivered but never arrived",
    "wrong item received in shipment",
    "missing items from my delivery",
    "change shipping address on order",
    "parcel arrived completely damaged",
    "delivery delayed past estimated date",
    "cancel order before shipping",
    # Cross-boundary ambiguous cases (Order vs Refund vs Billing)
    "Package arrived broken and damaged; should I request a return or file an insurance claim?",
    "I was charged for shipping order #{order} but tracking says it was cancelled by the merchant.",
    "Order #{order} has not arrived yet; do I have to wait before requesting a refund?",
    "Received wrong item in shipment; how do I exchange it without being charged again?",
    # Long & detailed
    "Dear Logistics Support, I placed order #{order} twelve days ago with guaranteed express 2-day delivery. The tracking status #{track} still shows 'In transit - departing distribution hub' since last Thursday with no subsequent scans. This package contains time-sensitive components for an event this Friday. Please locate this parcel immediately or dispatch a replacement.",
    "Hello, I am contacting you regarding shipment #{order}. The courier driver marked the delivery as signed by 'Resident', but our facility was closed for the national holiday and the security cameras confirm no courier vehicle arrived. Please open an investigation with the carrier."
]

REFUND_RETURNS_SEEDS = [
    # Returns, refunds, replacements
    "I would like to return the shoes from order #{order} because they do not fit my size.",
    "How can I generate and print a prepaid return shipping label for my RMA #{rma}?",
    "I returned the defective laptop two weeks ago (tracking #{track}); when will my refund be processed?",
    "The return portal says my item is past the 30-day window, but it was delivered only 12 days ago.",
    "I received a damaged coffee maker and I want a full refund back to my original payment card.",
    "Where is my credit note or bank refund for cancelled order #{order}?",
    "You deducted a $15 restocking fee from my refund, but the item was returned sealed and unopened.",
    "I was issued store credit instead of the direct bank refund I originally requested.",
    "Can I exchange my jacket for a larger size instead of doing a complete return and repurchase?",
    "The drop-off QR code for the return locker expired before I could bring the parcel.",
    "My refund status shows 'Processed' on your portal, but my bank has no record of incoming funds.",
    "I want to return an unwanted gift; can the refund be credited to my own personal account?",
    "The return courier lost my return package #{track}; am I still eligible for my refund?",
    "I cancelled my annual subscription within your 14-day money-back guarantee period; please refund me.",
    "Can I drop off my return package directly at your physical retail store instead of mailing it?",
    "Why did I only receive a partial refund of ${amt} when the total purchase price was ${amt2}?",
    "I returned three items in one box, but your warehouse only credited me for one item.",
    "Please send a replacement unit under warranty; the replacement process is taking too long.",
    "I want to return the product because it does not match the specifications advertised on your website.",
    "The return shipping label provided by your system has an invalid postal code for my region.",
    # Short variants
    "how to get prepaid return label",
    "refund not credited to my card",
    "want full refund for defective item",
    "status of return package",
    "exchange for a different size",
    "money back guarantee claim",
    "unauthorized restocking fee deducted",
    "cancel order and refund money",
    # Cross-boundary ambiguous cases (Refund vs Billing vs Order)
    "The charge on my card was not refunded after my returned package #{rma} was received.",
    "I cancelled order #{order} before shipping and want my payment refunded to my original card.",
    "My returned parcel was delivered back to your warehouse but my billing balance is not updated.",
    "Can you refund the monthly subscription since the software crashed and was unusable?",
    # Long & detailed
    "Customer Service, I initiated return RMA #{rma} on the 3rd and dropped off the package with the courier on the 5th (proof of drop-off tracking #{track}). The warehouse received the package six business days ago, yet my account dashboard still displays 'Awaiting return shipment'. I demand an update and immediate processing of my ${amt} refund.",
    "Hello, I purchased software license #{order} under your advertised 30-day no-questions-asked refund policy. After testing with our engineering infrastructure, the tool does not meet our compliance requirements. Please initiate a full refund to our corporate Visa ending in {card}."
]

GENERAL_INQUIRY_SEEDS = [
    # Information, policies, features, sales
    "What are your customer support telephone operating hours and time zones?",
    "Does your platform comply with GDPR, HIPAA, and SOC 2 Type II data privacy standards?",
    "Do you offer educational or non-profit organization discounts on annual subscriptions?",
    "Are your services and product delivery available to customers located in Australia and New Zealand?",
    "Where can I find documentation and code examples for your Python REST API SDK?",
    "Do you have a public product roadmap where users can suggest and upvote future features?",
    "Can I schedule a live product demonstration for our executive leadership team next Wednesday?",
    "What is the maximum file upload size supported on the Business tier versus Enterprise tier?",
    "Does your application support multilingual interfaces in Spanish, German, and Japanese?",
    "Where are your cloud data centers physically hosted (e.g. AWS US-East or EU-Central)?",
    "Can you provide a formal vendor security questionnaire for our procurement audit?",
    "What is the difference between the Starter plan and the Professional plan features?",
    "Do you have physical store locations or authorized distributors in the Chicago metropolitan area?",
    "How do I apply for your reseller partner affiliate program?",
    "Is it possible to migrate data directly from our existing CRM platform to your software?",
    "What encryption standards are used for customer data stored at rest and in transit?",
    "Can individual contractors sign up or do you require a registered legal business entity?",
    "Are there any upcoming webinars or training workshops for onboarding new administrators?",
    "What is your uptime Service Level Agreement (SLA) percentage for enterprise customers?",
    "Can you send me a product brochure and feature comparison matrix in PDF format?",
    # Short variants
    "what are your support working hours?",
    "do you offer student or non-profit discounts?",
    "is this platform GDPR compliant?",
    "where to find API documentation?",
    "schedule demo with sales team",
    "supported languages on mobile app",
    "enterprise tier pricing details",
    "data center locations for privacy",
    # Cross-boundary ambiguous cases (General vs Billing vs Tech)
    "What payment methods and currencies do you support for international annual invoices?",
    "Does your technical infrastructure support on-premise deployment or only cloud hosting?",
    "Do you offer a money-back refund guarantee if we test the enterprise subscription?",
    "Can you clarify the SLA uptime guarantee and how service outages affect monthly billing?",
    # Long & detailed
    "Good day, Our procurement department is evaluating your SaaS platform for enterprise deployment across 250 workstations. Could you provide your standard Service Level Agreement (SLA) documentation, SOC 2 compliance reports, and schedule an introductory technical discovery call with your solutions architect?",
    "Hello team, I am an independent researcher evaluating various NLP and data analytics platforms for an upcoming academic case study. Could you clarify whether your platform allows API data extraction for academic research and if special educational tier licensing is available?"
]

# Real-world typographical errors & informal variants for diversity
TYPO_REPLACEMENTS = {
    "receive": "recieve",
    "received": "recieved",
    "account": "acount",
    "password": "pasword",
    "subscription": "subcription",
    "payment": "paymnt",
    "refund": "refnd",
    "cancel": "cancl",
    "cancelled": "canceled",
    "delivery": "delivry",
    "address": "adress",
    "package": "pakage",
    "invoice": "invoce",
    "please": "plz",
    "thanks": "thx",
    "does not": "dosent",
    "cannot": "cant",
    "could not": "cudnt",
    "immediately": "immediatly"
}

def inject_typos_and_noise(text: str, prob: float = 0.25) -> str:
    """Randomly inject realistic typos or casing anomalies into a sentence."""
    if random.random() > prob:
        return text
    
    words = text.split()
    new_words = []
    for w in words:
        lower_w = w.lower().strip(".,!?:;")
        punct = w[len(lower_w):] if len(w) > len(lower_w) else ""
        if lower_w in TYPO_REPLACEMENTS and random.random() < 0.4:
            new_w = TYPO_REPLACEMENTS[lower_w]
            new_words.append(new_w + punct)
        else:
            new_words.append(w)
    
    result = " ".join(new_words)
    # Occasionally lowercase everything
    if random.random() < 0.15:
        result = result.lower()
    return result

def populate_templates(templates, target_count=260):
    """
    Generate realistic variants from seed templates using rich combinatorial parameters.
    """
    variants = set()
    
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Cameron", "Jesse"]
    domains = ["example.com", "corporate.org", "techfirm.io", "startup.co", "globalbiz.net"]
    
    while len(variants) < target_count:
        tpl = random.choice(templates)
        
        # Combinatorial substitutions
        amt = random.randint(15, 850)
        amt2 = amt + random.randint(20, 150)
        inv = random.randint(10045, 99982)
        tx = f"TXN{random.randint(100000, 999999)}"
        card = f"{random.randint(1000, 9999)}"
        code = random.choice(["SAVE20", "WINTER50", "WELCOME10", "BLACKFRIDAY", "SPRING25", "DISCOUNT15"])
        acct = f"ACC-{random.randint(1000, 9999)}"
        job = f"JOB-{random.randint(4000, 8999)}"
        user = f"usr_{random.randint(100, 999)}"
        email = f"{random.choice(first_names).lower()}{random.randint(1, 99)}@{random.choice(domains)}"
        email2 = f"new.{random.choice(first_names).lower()}@{random.choice(domains)}"
        order = f"ORD-{random.randint(20000, 89999)}"
        track = f"TRK{random.randint(10000000, 99999999)}"
        rma = f"RMA-{random.randint(10000, 49999)}"
        
        text = tpl.replace("{amt}", str(amt))\
                  .replace("{amt2}", str(amt2))\
                  .replace("{inv}", str(inv))\
                  .replace("{tx}", tx)\
                  .replace("{card}", card)\
                  .replace("{code}", code)\
                  .replace("{acct}", acct)\
                  .replace("{job}", job)\
                  .replace("{user}", user)\
                  .replace("{email}", email)\
                  .replace("{email2}", email2)\
                  .replace("{order}", order)\
                  .replace("{track}", track)\
                  .replace("{rma}", rma)
        
        # Add random greetings or sign-offs occasionally
        prefixes = [
            "",
            "Hi, ",
            "Hello, ",
            "Urgent: ",
            "Dear Support, ",
            "Hey team, ",
            "Good morning, ",
            "Help needed: ",
            "Please assist: "
        ]
        suffixes = [
            "",
            " Please help as soon as possible.",
            " Thanks in advance.",
            " Awaiting your swift response.",
            " Let me know how to resolve this.",
            " Thank you.",
            " This is blocking our business.",
            " Please reply promptly."
        ]
        
        if random.random() < 0.35 and not text.startswith("Hello") and not text.startswith("Dear"):
            text = random.choice(prefixes) + text
        if random.random() < 0.35 and not text.endswith("Thank you."):
            text = text + random.choice(suffixes)
            
        # Apply typo / casing augmentation
        text = inject_typos_and_noise(text, prob=0.30)
        
        clean_text = " ".join(text.split())
        if len(clean_text) > 8:
            variants.add(clean_text)
            
    return list(variants)[:target_count]

def generate_full_dataset():
    data = []
    
    seeds_map = {
        "Billing": BILLING_SEEDS,
        "Technical Support": TECH_SUPPORT_SEEDS,
        "Account Access": ACCOUNT_ACCESS_SEEDS,
        "Order Issues": ORDER_ISSUES_SEEDS,
        "Refund/Returns": REFUND_RETURNS_SEEDS,
        "General Inquiry": GENERAL_INQUIRY_SEEDS
    }
    
    ticket_counter = 10001
    
    for category, seeds in seeds_map.items():
        tickets = populate_templates(seeds, target_count=260)
        for t in tickets:
            data.append({
                "ticket_id": f"TCK-{ticket_counter}",
                "text": t,
                "category": category
            })
            ticket_counter += 1
            
    df = pd.DataFrame(data)
    
    # Check duplicates and clean
    before_dedup = len(df)
    # Deduplicate by normalized text
    df["norm_text"] = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df = df.drop_duplicates(subset=["norm_text"]).reset_index(drop=True)
    df = df.drop(columns=["norm_text"])
    after_dedup = len(df)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    
    # Re-assign sequential clean IDs
    df["ticket_id"] = [f"TCK-{10001 + i}" for i in range(len(df))]
    
    print(f"Generated {before_dedup} raw tickets. After deduplication: {after_dedup} distinct tickets.")
    print("Class distribution:")
    print(df["category"].value_counts())
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "tickets.csv")
    df = generate_full_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to: {out_path}")
