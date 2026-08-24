# Enterprise RAG Sample Q&A Log

> Ground-truth evaluation record featuring 15 benchmark questions across 4 test categories (including 4 unanswerable out-of-domain queries) as required by Assignment Section 6.

## Performance Scorecard

| Benchmark Metric | Measured Score | Requirement | Evaluation Status |
|---|---|---|---|
| **Retrieval Recall@5** | **`100.0%`** | $\ge 90\%$ | 🟢 PASS |
| **Citation Precision** | **`100.0%`** | $\ge 90\%$ | 🟢 PASS |
| **Abstention Precision** | **`100.0%`** | $\ge 95\%$ | 🟢 PASS |
| **Abstention Recall** | **`100.0%`** | $\ge 95\%$ | 🟢 PASS |
| **Abstention F1-Score** | **`1.000`** | $\ge 0.95$ | 🟢 PASS |
| **Mean Latency** | **`2476.5 ms`** | $< 3500\text{ms}$ | 🟢 PASS |

---

## Evaluated Questions & Grounded Responses

### 📌 Category 1: Direct In-Domain Factual Q&A

#### Case #01: What are the storage capacities and RAID configurations of the CSP-200 and CSP-400 models?
- **Intent / Target:** Verifies direct extraction of hardware models and storage capacity from Product Manual.
- **Expected Source:** `Product_Manual` | **Expected Facts:** `['4TB', '16TB', 'RAID']`
- **Confidence:** `0.843` (HIGH) | **Latency:** `2400.7 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> The CSP‑200 model provides **4 TB of usable storage configured with RAID 1**.  
> The CSP‑400 model provides **16 TB of usable storage configured with RAID 5**.  
> 
> These specifications are stated in the troubleshooting section of the product manual:
> 
> - “The CSP‑200 ships with 4TB of usable storage (RAID 1); the CSP‑400 ships with 16TB usable (RAID 5).” [Product_Manual (PM-CSP-001 v3.2), Page 3]

**Attributed Provenance Citations:**
- `[Product_Manual (PM-CSP-001 v3.2), Page 3]` (Page 3)
- `[Product_Manual (PM-CSP-001 v3.2), Page 2]` (Page 2)
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]` (Page 2)
- `[Product_Manual (PM-CSP-001 v3.2), Page 1]` (Page 1)

---

#### Case #02: What authentication header and token format are required for API requests?
- **Intent / Target:** Verifies API header extraction and Bearer token requirement from API Reference.
- **Expected Source:** `API_Reference` | **Expected Facts:** `['Bearer', 'Authorization']`
- **Confidence:** `0.847` (HIGH) | **Latency:** `2830.6 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> All API calls must include an **Authorization header** whose value is a **Bearer token** (e.g., `Authorization: Bearer <access_token>`). The token is obtained from the `/oauth/token` endpoint and is required for every request.  
> 
> [API_Reference (API-REF-002 v3.2), Page 2]

**Attributed Provenance Citations:**
- `[API_Reference (API-REF-002 v3.2), Page 2]` (Page 2)
- `[API_Reference (API-REF-002 v3.2), Page 3]` (Page 3)
- `[FAQ_Support (FAQ-SUP-014 v3.2), Page 2]` (Page 2)

---

#### Case #03: How much Paid Time Off (PTO) do full-time employees accrue per month according to the leave policy?
- **Intent / Target:** Verifies HR policy PTO accrual rate from Employee Handbook.
- **Expected Source:** `Employee_Handbook` | **Expected Facts:** `['1.75 days', 'PTO']`
- **Confidence:** `0.883` (HIGH) | **Latency:** `2044.8 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> Full‑time employees accrue **1.75 days of Paid Time Off (PTO) each month** [Employee_Handbook (HR-EH-2026 v3.2), Page 2].

**Attributed Provenance Citations:**
- `[Employee_Handbook (HR-EH-2026 v3.2), Page 2]` (Page 2)
- `[Onboarding_Guide (ONB-GDE-009 v3.2), Page 3]` (Page 3)

---

#### Case #04: Within what timeframe must security incidents involving Confidential or Restricted data be notified to clients?
- **Intent / Target:** Verifies incident notification window from Security Policy.
- **Expected Source:** `Security_Policy` | **Expected Facts:** `['72 hours']`
- **Confidence:** `0.866` (HIGH) | **Latency:** `2327.0 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> Security incidents that involve Confidential or Restricted data must be communicated to affected clients within **72 hours** of discovery.  
> 
> [Security_Policy (SEC-POL-007 v3.2), Page 2]

**Attributed Provenance Citations:**
- `[Security_Policy (SEC-POL-007 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 3]` (Page 3)
- `[Security_Policy (SEC-POL-007 v3.2), Page 1]` (Page 1)

---

#### Case #05: How many days before their first day will a new employee receive laptop shipping confirmation?
- **Intent / Target:** Verifies pre-start equipment shipping timeline from Onboarding Guide.
- **Expected Source:** `Onboarding_Guide` | **Expected Facts:** `['5 days']`
- **Confidence:** `0.876` (HIGH) | **Latency:** `2330.8 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> A new employee will receive the laptop shipping confirmation **5 business days before their start date**【Onboarding_Guide (ONB-GDE-009 v3.2), Page 3】.

**Attributed Provenance Citations:**
- `[Onboarding_Guide (ONB-GDE-009 v3.2), Page 3]` (Page 3)
- `[Onboarding_Guide (ONB-GDE-009 v3.2), Page 1]` (Page 1)
- `[Onboarding_Guide (ONB-GDE-009 v3.2), Page 2]` (Page 2)

---

### 📊 Category 2: 2D Table Matrix Reasoning

#### Case #06: What are the monthly subscription costs and storage limits for Free and Standard tiers?
- **Intent / Target:** Verifies 2D matrix extraction across subscription pricing columns.
- **Expected Source:** `Pricing_and_SLA` | **Expected Facts:** `['$0', '5 GB', '$12', '100 GB']`
- **Confidence:** `0.904` (HIGH) | **Latency:** `2929.6 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> **Free tier**  
> - Monthly subscription cost: **$0**  
> - Storage limit: **5 GB**  
> 
> **Standard tier**  
> - Monthly subscription cost: **$12 per user per month**  
> - Storage limit: **500 GB pooled**  
> 
> These figures are taken directly from the pricing table for the Free and Standard tiers【Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2】.

**Attributed Provenance Citations:**
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]` (Page 2)
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 1]` (Page 1)

---

#### Case #07: What is the guaranteed SLA uptime percentage for Standard vs Enterprise tiers?
- **Intent / Target:** Verifies SLA tiering uptime table extraction from Pricing & SLA document.
- **Expected Source:** `Pricing_and_SLA` | **Expected Facts:** `['99.9%', '99.99%']`
- **Confidence:** `0.861` (HIGH) | **Latency:** `4100.2 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> - **Standard tier:** The SLA guarantees **99.5 % monthly uptime**.  
>   [Pricing_and_SLA (PRC‑SLA‑021 v3.2), Page 2]
> 
> - **Enterprise tier:** The SLA guarantees **99.95 % monthly uptime**.  
>   [Pricing_and_SLA (PRC‑SLA‑021 v3.2), Page 2]

**Attributed Provenance Citations:**
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]` (Page 2)
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 1]` (Page 1)

---

#### Case #08: At what time does the Nightly backup schedule run according to the Product Manual backup table?
- **Intent / Target:** Verifies scheduled backup timing extraction from Product Manual table.
- **Expected Source:** `Product_Manual` | **Expected Facts:** `['2:00 AM', 'Nightly']`
- **Confidence:** `0.652` (MEDIUM) | **Latency:** `6374.1 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> The Nightly backup schedule runs **once daily at 2:00 AM local time**.  
> 
> [Product_Manual (PM-CSP-001 v3.2), Page 2]

**Attributed Provenance Citations:**
- `[Product_Manual (PM-CSP-001 v3.2), Page 2]` (Page 2)
- `[Onboarding_Guide (ONB-GDE-009 v3.2), Page 3]` (Page 3)
- `[Product_Manual (PM-CSP-001 v3.2), Page 1]` (Page 1)

---

### CROSS_DOC

#### Case #09: Is two-factor authentication (2FA) required for Free and Standard tiers, or is it mandatory for Enterprise?
- **Intent / Target:** Verifies 2FA policy across subscription tiers.
- **Expected Source:** `FAQ_Support, Pricing_and_SLA` | **Expected Facts:** `['optional', 'Free', 'Standard', 'Enterprise']`
- **Confidence:** `0.966` (HIGH) | **Latency:** `2059.7 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> Two‑factor authentication (2FA) is **optional** for Free and Standard tier accounts, but it is **mandatory** for all Enterprise tier accounts (and for any account that has admin‑level permissions)【FAQ_Support (FAQ-SUP-014 v3.2), Page 2】.

**Attributed Provenance Citations:**
- `[FAQ_Support (FAQ-SUP-014 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 2]` (Page 2)
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]` (Page 2)
- `[Employee_Handbook (HR-EH-2026 v3.2), Page 1]` (Page 1)

---

#### Case #10: How long is customer data retained if a user cancels their subscription?
- **Intent / Target:** Verifies cancellation data retention period.
- **Expected Source:** `FAQ_Support, Pricing_and_SLA` | **Expected Facts:** `['30 days', 'retained']`
- **Confidence:** `0.913` (HIGH) | **Latency:** `1912.0 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> When a subscription is cancelled, Atman Cloud keeps the customer’s data in a read‑only state for **90 days** after the cancellation. After this 90‑day window the data is permanently deleted, though the user can export it at any time during the retention period via **Account > Export**.  
> 
> **Citation:** FAQ_Support (FAQ-SUP-014 v3.2), Page 2.

**Attributed Provenance Citations:**
- `[FAQ_Support (FAQ-SUP-014 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 3]` (Page 3)
- `[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 2]` (Page 2)

---

#### Case #11: How can a user reset their password, and for how long is the reset link valid?
- **Intent / Target:** Verifies FAQ password reset procedure and token expiration window.
- **Expected Source:** `FAQ_Support` | **Expected Facts:** `['Forgot password', '15 minutes']`
- **Confidence:** `0.892` (HIGH) | **Latency:** `4699.0 ms` | **Status:** 🟢 [ANSWERED]

**Grounded System Response:**
> A user can reset their password by going to the login page and clicking **“Forgot password.”** An email containing a reset link is sent to the user’s registered address, and that link remains valid for **24 hours** before it expires.  
> 
> **Citation:** [FAQ_Support (FAQ-SUP-014 v3.2), Page 2]

**Attributed Provenance Citations:**
- `[FAQ_Support (FAQ-SUP-014 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 2]` (Page 2)
- `[Security_Policy (SEC-POL-007 v3.2), Page 3]` (Page 3)
- `[API_Reference (API-REF-002 v3.2), Page 2]` (Page 2)

---

### 🛑 Category 4: Out-of-Domain Abstention & Guardrail Validation (Unanswerable)

#### Case #12: What is the authentic Italian recipe and oven temperature for baking chocolate fudge cake?
- **Intent / Target:** Trap query: Culinary topic unmentioned in corpus. Must trigger Layer 1 abstention.
- **Expected Source:** `` | **Expected Facts:** `[]`
- **Confidence:** `0.040` (ABSTAINED) | **Latency:** `872.9 ms` | **Status:** 🔴 [ABSTAINED]

**Grounded System Response:**
> Insufficient document evidence (confidence 0.040 < 0.400). This question may be outside the scope of the provided documents.

---

#### Case #13: Who won the Premier League football championship and who was the top goal scorer last season?
- **Intent / Target:** Trap query: Sports trivia unmentioned in corpus. Must trigger Layer 1 abstention.
- **Expected Source:** `` | **Expected Facts:** `[]`
- **Confidence:** `0.041` (ABSTAINED) | **Latency:** `865.1 ms` | **Status:** 🔴 [ABSTAINED]

**Grounded System Response:**
> Insufficient document evidence (confidence 0.041 < 0.400). This question may be outside the scope of the provided documents.

---

#### Case #14: What is the 7-day weather forecast, humidity, and rainfall probability for Tokyo, Japan?
- **Intent / Target:** Trap query: Meteorology topic unmentioned in corpus. Must trigger Layer 1 abstention.
- **Expected Source:** `` | **Expected Facts:** `[]`
- **Confidence:** `0.096` (ABSTAINED) | **Latency:** `580.9 ms` | **Status:** 🔴 [ABSTAINED]

**Grounded System Response:**
> Insufficient document evidence (confidence 0.096 < 0.400). This question may be outside the scope of the provided documents.

---

#### Case #15: Can you give me a full biography, discography, and Grammy awards list for Taylor Swift?
- **Intent / Target:** Trap query: Pop culture unmentioned in corpus. Must trigger Layer 1 abstention.
- **Expected Source:** `` | **Expected Facts:** `[]`
- **Confidence:** `0.053` (ABSTAINED) | **Latency:** `820.3 ms` | **Status:** 🔴 [ABSTAINED]

**Grounded System Response:**
> Insufficient document evidence (confidence 0.053 < 0.400). This question may be outside the scope of the provided documents.

---
