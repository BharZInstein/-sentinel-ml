# SentinelML

### A Runtime Adversarial Attack Detection & Defense Platform for Production ML Systems

**Technical Design Document & Research Proposal**

| | |
|---|---|
| **Document Type** | Technical Design Document (TDD) + Research Proposal |
| **Project Codename** | SentinelML |
| **Classification** | AI/ML Security — Runtime Defense |
| **Team Size** | 3 (Research + Engineering) |
| **Timeline** | 8 Weeks |
| **Version** | 1.0 |
| **Date** | August 17, 2026 |

---

## Table of Contents

1. [Executive Summary & Objective](#1-executive-summary--objective)
2. [Detailed Problem Statement](#2-detailed-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [8-Week Project Milestones](#6-8-week-project-milestones)
7. [Risks, Ethics & Limitations](#7-risks-ethics--limitations)
8. [References](#8-references)

---

## 1. Executive Summary & Objective

### 1.1 Summary

Machine learning models are increasingly deployed as network-accessible services, yet they are shipped with essentially **no runtime security layer**. A production image classifier or Large Language Model (LLM) will faithfully process a maliciously perturbed image or an injected prompt with the same confidence it applies to benign input. The security posture of most ML pipelines today is equivalent to running a web server with no firewall, no WAF, and no intrusion detection system (IDS).

**SentinelML** is a full-pipeline defense platform that functions as an **AI Firewall / API Gateway** sitting in front of a protected ("target") model. Every inbound inference request is intercepted, screened by a dedicated **detector model** trained to distinguish clean inputs from adversarially manipulated ones, and either forwarded to the target model or blocked and logged. Conceptually, SentinelML is to an ML inference endpoint what a network IDS/IPS is to a corporate network: a transparent, real-time guard layer that observes traffic, scores threat likelihood, and enforces policy.

The platform spans two threat domains that are usually studied in isolation:

- **Vision evasion attacks** — gradient-based perturbations such as **FGSM** and **PGD** that cause misclassification while remaining imperceptible to humans.
- **LLM manipulation** — **prompt injection**, **jailbreak** patterns, and **data-poisoning** simulations that subvert instruction-following.

### 1.2 Research Objective

Most published adversarial-detection work is **attack-specific**: a detector is trained and evaluated against a single attack (e.g., PGD at a fixed epsilon) and reports high accuracy in that narrow setting. This produces brittle defenses and inflated confidence.

> **Core research question:** *To what extent can a single detector model generalize across heterogeneous attack families — and how quickly does that generalization degrade under an adaptive, white-box adversary who can see the detector?*

SentinelML's contribution is therefore **not** claiming a state-of-the-art detector. It is an **honest, reproducible benchmark** of cross-attack generalization and adaptive-attack resilience, including a rigorous report of *where and how the detector fails*. Negative and partial results are treated as first-class findings.

### 1.3 Success Criteria

| Objective | Target Outcome |
|---|---|
| **Functional gateway** | Stateless proxy adds < 50 ms median overhead per request |
| **Detection (in-distribution)** | ≥ 90% detection accuracy on attacks seen during training |
| **Generalization (out-of-distribution)** | Quantified detection accuracy on *held-out* attack families, reported without cherry-picking |
| **Clean traffic safety** | False-Positive Rate (FPR) ≤ 5% on benign production-like traffic |
| **Adaptive robustness** | Documented degradation curve under white-box adaptive attacks (the "arms race" report) |

---

## 2. Detailed Problem Statement

### 2.1 The Runtime Security Gap

Contemporary ML security research is dominated by **training-time hardening** — adversarial training, certified robustness, gradient masking. These approaches share three practical limitations:

1. **They require retraining the target model.** Most teams consume models they did not train (foundation models, vendor APIs, fine-tuned checkpoints) and cannot re-harden them.
2. **They trade clean accuracy for robustness.** Adversarial training reliably degrades performance on benign inputs — a cost product teams refuse to pay.
3. **They are static.** A model hardened at training time cannot adapt to attack techniques discovered after deployment.

Meanwhile, the **deployment surface** — the live API endpoint where attacks actually arrive — has almost no dedicated defensive tooling. There is no equivalent of Snort, Suricata, or a Web Application Firewall for inference traffic. **SentinelML targets this runtime gap specifically**: a detachable, model-agnostic layer that protects models it did not train and does not modify.

### 2.2 The Generalization Problem

The central weakness in the adversarial-detection literature is **overfitting to the threat model**. A representative failure mode:

- A detector is trained on PGD examples at `ε = 8/255`.
- It reports 98% detection accuracy — against PGD at `ε = 8/255`.
- The same detector is near-random against FGSM, against PGD at a different epsilon, against a spatial-transformation attack, or against any attack it never saw.

This is the ML-security analogue of a signature-based antivirus that only recognizes known malware hashes. In a runtime setting, attackers are not obligated to use the attack you trained against. **A defense that does not generalize is a defense that provides false assurance** — arguably worse than no defense, because it induces misplaced trust.

SentinelML confronts this directly by:

- Training on a **mixture** of attack families and holding out others for out-of-distribution (OOD) evaluation.
- Reporting a **generalization matrix** (train-attack × test-attack) rather than a single headline number.
- Refusing to report only the favorable diagonal of that matrix.

### 2.3 The Adaptive Adversary Problem

A defense evaluated only against **static, detector-unaware attacks** is evaluated under the weakest possible threat model. The stronger and more realistic threat is the **adaptive white-box adversary** (Athalye et al., 2018; Tramèr et al., 2020) who knows the detector exists and optimizes attacks against *both* the target model and the detector simultaneously.

Historically, most detection defenses that reported success against static attacks were **broken within months** by adaptive attacks. Any credible modern detection paper must therefore include an adaptive-attack evaluation. SentinelML's red-team module (Section 5.3) makes this the centerpiece rather than an afterthought, and reports the resulting **arms race honestly** — including the point at which the detector is defeated.

### 2.4 Scope Statement

**In scope:** runtime detection and blocking; FGSM/PGD vision evasion; prompt injection / jailbreak / poisoning simulation for LLMs; cross-attack generalization; adaptive red-teaming; latency and logging.

**Explicitly out of scope (for this 8-week effort):** training-time robustification of the target model; certified/provable robustness guarantees; model-extraction and membership-inference defenses; production-grade multi-tenant authentication and horizontal autoscaling (a single-node reference deployment is delivered).

---

## 3. System Architecture

### 3.1 Design Principles

- **Transparent interception.** Clients target SentinelML's endpoint with the *same request schema* the target model expects. No client-side changes are required.
- **Stateless screening.** Each request is scored independently; no per-request session state is required for the detection decision. This keeps the gateway horizontally scalable and simple to reason about.
- **Fail-open vs. fail-closed is a policy, not a default.** The operator configures whether an internal detector error blocks or forwards traffic. Security-critical deployments choose fail-closed.
- **Observability first.** Every decision — allow, block, or error — is logged with enough context to reconstruct an attack timeline in the dashboard.

### 3.2 End-to-End Traffic Flow

```
                        ┌──────────────────────────────────────────────────────┐
                        │                    SentinelML Gateway                  │
                        │                                                        │
  ┌────────┐   (1)      │  ┌───────────┐   (2)      ┌──────────────┐             │
  │        │  request   │  │           │ normalize  │              │             │
  │ Client │───────────►│─►│ API Proxy │───────────►│  Detector    │             │
  │        │            │  │ (Ingress) │            │  Model       │             │
  └────────┘            │  └───────────┘            │ (clean/adv?) │             │
      ▲                 │        │                   └──────┬───────┘             │
      │                 │        │                          │ (3) score          │
      │                 │        │            ┌─────────────┴─────────────┐      │
      │                 │        │            │                           │      │
      │                 │        │       score < τ                   score ≥ τ   │
      │                 │        │      (BENIGN)                     (ADVERSARIAL)│
      │                 │        │            │                           │      │
      │                 │        ▼            ▼                           ▼      │
      │                 │  ┌──────────┐  ┌─────────┐              ┌────────────┐ │
      │                 │  │ Logging  │◄─┤ Forward │              │  Block +   │ │
      │                 │  │ / SIEM   │  │ to Target│             │  Log Event │ │
      │                 │  └────┬─────┘  └────┬────┘              └─────┬──────┘ │
      │                 │       │             │ (4)                     │        │
      │                 └───────┼─────────────┼─────────────────────────┼────────┘
      │                         │             ▼                         │
      │                         │      ┌──────────────┐                 │
      │                         │      │ Target Model │ (5)             │
      │                         │      │ (Protected)  │                 │
      │                         │      └──────┬───────┘                 │
      │                         │             │                         │
      │   (6) inference result  │             │        (6') 403 blocked │
      └─────────────────────────┴─────────────┴─────────────────────────┘
```

### 3.3 Step-by-Step Traffic Description

1. **Client → API Proxy (Ingress).** The client sends a standard inference request (an image payload or a text prompt) to the SentinelML endpoint, which mirrors the target model's public contract.
2. **Proxy → Detector (Normalization + Feature Extraction).** The proxy normalizes the payload (decoding, resizing, tokenization) and passes it to the detector. Normalization is deliberately *identical* to the target model's preprocessing so the detector sees exactly what the target would.
3. **Detector Scoring.** The detector emits an adversarial-likelihood score `s ∈ [0, 1]`. A configurable threshold `τ` converts the score into a decision.
4. **Decision Branch.**
   - `s < τ` → **BENIGN**: the request is forwarded, unmodified, to the target model.
   - `s ≥ τ` → **ADVERSARIAL**: the request is blocked (HTTP `403`), the target model is never contacted, and a structured attack event is logged.
5. **Target Model Inference.** For benign traffic only, the target model produces its normal prediction/response.
6. **Response.** The client receives either the target model's genuine inference result (`6`) or a policy-driven block response (`6'`). In both cases the event is recorded for the dashboard.

### 3.4 Component Inventory

| Component | Responsibility | Reference Technology |
|---|---|---|
| **API Proxy / Ingress** | TLS termination, request normalization, routing, threshold policy | FastAPI / Uvicorn (ASGI) |
| **Detector Service** | Adversarial scoring; hot-swappable model | PyTorch + TorchServe *or* in-process module |
| **Target Model Adapter** | Uniform interface to any downstream model (vision or LLM) | Pluggable adapter class |
| **Event Store / SIEM** | Append-only log of every decision | SQLite → PostgreSQL |
| **Dashboard** | Real-time attack visualization, FPR/TPR monitoring | Streamlit / React + FastAPI |

---

## 4. Module Breakdown

### 4.1 Data & Attack Generation Pipeline

**Purpose.** Produce a large, balanced, *labeled* corpus of `{clean, adversarial}` inputs spanning multiple attack families, and split it so that some families are reserved for out-of-distribution generalization testing.

#### 4.1.1 Vision Attack Generation

- **Base datasets:** CIFAR-10 and a subset of ImageNet for the target classifier.
- **Attacks:**
  - **FGSM** (Fast Gradient Sign Method) — single-step, `ε`-bounded perturbation.
  - **PGD** (Projected Gradient Descent) — multi-step, iterative; the standard strong first-order attack. Generated across a **range** of `ε` values and step counts to prevent single-configuration overfitting.
  - *(Stretch)* **C&W** and a spatial/rotation attack, reserved as **held-out OOD families**.
- **Tooling:** established libraries (e.g., Foolbox, CleverHans, or `torchattacks`) for reproducibility and to avoid reimplementation bugs.

#### 4.1.2 LLM Attack Generation

- **Prompt injection:** direct ("ignore previous instructions…") and indirect (payloads embedded in retrieved/quoted content).
- **Jailbreak patterns:** role-play, obfuscation/encoding, and known template families curated from public red-team datasets.
- **Data poisoning simulation:** construction of poisoned fine-tuning samples to demonstrate the *training-time* threat and to seed the detector with poisoned-input signatures.

#### 4.1.3 Dataset Governance

- **Balance:** roughly 50/50 clean vs. adversarial to avoid base-rate bias.
- **Provenance labels:** each example carries its attack family, hyperparameters, and source seed for full traceability.
- **The critical split:**

| Split | Attack families | Use |
|---|---|---|
| **Train** | e.g., FGSM + PGD (subset of ε) | Detector training |
| **In-distribution test** | Same families, unseen seeds | Sanity ceiling |
| **Out-of-distribution test** | **Held-out families** (e.g., C&W, spatial, novel jailbreaks) | **Generalization measurement — the core result** |

### 4.2 The Detector Model (Training & Generalization Strategy)

**Purpose.** A single model that outputs an adversarial-likelihood score, trained to generalize *beyond* the specific attacks it was trained on.

#### 4.2.1 Modeling Approaches (evaluated comparatively)

1. **Supervised binary classifier** operating on the input (or the target model's intermediate feature activations). Feature-space detection is often more robust than pixel-space detection.
2. **Statistical / distributional detection** — e.g., detecting inputs that fall off the natural-data manifold (feature-squeezing, kernel-density, or Mahalanobis-distance style scoring). These are frequently *more* transferable across attacks and serve as strong generalization baselines.
3. **For LLMs:** a lightweight classifier over embeddings plus curated heuristic/regex signatures for known injection templates (a hybrid signature + learned detector, mirroring modern IDS design).

#### 4.2.2 Generalization Strategy (the research core)

- **Attack mixing:** train on multiple families and epsilons simultaneously.
- **Leave-one-attack-out (LOAO) protocol:** systematically hold out each attack family in turn and measure detection on the excluded family. This yields the **generalization matrix** and is the paper's headline evidence.
- **Feature-space over input-space:** prefer detectors operating on target-model activations, which empirically transfer better across attacks.
- **Honest failure reporting:** where the detector collapses on a held-out family, that is documented as a finding, not hidden.

#### 4.2.3 Output Contract

```json
{
  "score": 0.87,
  "decision": "block",
  "threshold": 0.5,
  "attack_family_guess": "pgd_like",
  "detector_version": "v1.2",
  "latency_ms": 11.4
}
```

### 4.3 The Middleware Gateway (Latency, Stateless Interception, Logging)

**Purpose.** The production-facing enforcement layer that makes detection usable in real time.

#### 4.3.1 Latency Budget

- The gateway adds a detection hop, so overhead must be tightly bounded. **Target: < 50 ms median added latency.**
- Techniques: detector runs on the same accelerator as, or co-located with, the target; batching for throughput; asynchronous, non-blocking logging (the log write is never on the critical path).
- Latency is **measured and reported** (p50/p95/p99), because an accurate-but-slow guard is operationally useless.

#### 4.3.2 Stateless Interception

- Each request's allow/block decision depends only on that request, enabling trivial horizontal scaling behind a load balancer.
- The proxy preserves the target model's request/response schema so it is a **drop-in** in front of an existing endpoint.
- **Policy configuration:** threshold `τ`, and **fail-open vs. fail-closed** on detector error, are operator-set.

#### 4.3.3 Logging & Observability (the "SIEM for ML")

- **Append-only event log** capturing timestamp, source metadata, decision, score, detector version, and (for blocked events) a payload reference for forensic review.
- **Dashboard** surfaces: live attack rate, attack-family distribution, rolling FPR/TPR, latency percentiles, and a per-request drill-down — an ML-pipeline analogue of a security SIEM console.
- **Alerting hook** (stretch): threshold-crossing on attack rate triggers a notification.

---

## 5. Evaluation Methodology

Evaluation is the scientific heart of SentinelML. Results are reported on the **held-out OOD split** and under **adaptive attack**, never only on the easy in-distribution case.

### 5.1 Detection Accuracy

- **Primary metrics:** Detection Rate (True-Positive Rate / Recall on adversarial inputs) and overall accuracy on the balanced test set.
- **Threshold-independent metric:** **ROC-AUC**, so results do not depend on a single hand-picked `τ`.
- **The generalization matrix:** a train-family × test-family table of detection rates. The **off-diagonal** entries (train on A, test on B) are the real result; the on-diagonal entries are the (expected-high) sanity ceiling.

| Trained on ↓ / Tested on → | FGSM | PGD | C&W (held-out) | Spatial (held-out) |
|---|---|---|---|---|
| **FGSM** | high (ceiling) | ? | ? | ? |
| **PGD** | ? | high (ceiling) | ? | ? |
| **FGSM + PGD** | high | high | **? (key result)** | **? (key result)** |

### 5.2 False-Positive Rate on Clean Traffic

> An oversensitive guard is worse than no guard: it blocks paying users and trains operators to disable it.

- **Metric:** FPR = fraction of *clean* inputs incorrectly blocked. **Target ≤ 5%.**
- **Realistic clean corpus:** clean traffic must include *hard* benign inputs — naturally noisy, compressed, low-quality, or out-of-domain-but-legitimate samples — not just pristine test images. FPR measured only on clean textbook data is meaningless.
- **Operating-point analysis:** report the full ROC curve and explicitly state the chosen `τ` and the TPR/FPR trade-off it implies.

### 5.3 Red-Teaming & Adaptive Attacks (the Arms Race)

This is the module that separates a credible defense from a fragile one.

- **Threat model:** **white-box adaptive adversary** with full knowledge of the detector's architecture and weights, optimizing perturbations against the **combined** (target + detector) system — the standard set by Athalye et al. (2018) and Tramèr et al. (2020).
- **Method:** the team attacks its *own* detector, crafting inputs that fool the target model while keeping the detector score below `τ`.
- **Reported outcomes (all of them, honestly):**
  1. The FPR/TPR degradation curve as adaptive-attack strength increases.
  2. The point at which the detector is defeated (near-zero detection at bounded perturbation), if reached.
  3. Any observed **cost imposed on the attacker** — e.g., the minimum perturbation budget required to simultaneously evade the detector, which may push perturbations toward human-perceptibility. *Raising the attacker's cost is a legitimate defensive win even when perfect detection is impossible.*
- **Deliverable:** a candid "arms race" narrative — round by round — rather than a single triumphant number.

### 5.4 Ablations & Baselines

- Compare the learned detector against simple baselines (input-preprocessing / feature-squeezing) to show the learned model earns its complexity.
- Ablate input-space vs. feature-space detection to substantiate the generalization strategy claims.

---

## 6. 8-Week Project Milestones

Roles are assigned by center of gravity; all three members review each other's work. Suggested roles:

- **Member A — Research & Attacks (Adversarial ML lead)**
- **Member B — Detector Modeling (ML training lead)**
- **Member C — Systems & Gateway (Infra / full-stack lead)**

| Week | Phase | Member A — Attacks & Research | Member B — Detector | Member C — Gateway & Systems |
|---|---|---|---|---|
| **1** | Foundations | Literature review; finalize threat model & attack list | Set up training infra; pick target vision model | Scaffold FastAPI proxy; define request/response contract |
| **2** | Data Pipeline | Implement FGSM/PGD generation; begin LLM injection corpus | Baseline detector on FGSM only; establish metrics harness | Stateless interception + pass-through to a dummy target |
| **3** | First Detector | Generate balanced clean/adv dataset v1; define train/OOD split | Train mixed-attack detector; first generalization matrix | Integrate detector into proxy; measure baseline latency |
| **4** | Integration (Milestone: **end-to-end demo**) | LLM attack corpus complete | Feature-space vs. input-space comparison | Logging/event store online; skeleton dashboard |
| **5** | Generalization | Curate held-out OOD families (C&W, spatial, novel jailbreaks) | Leave-one-attack-out study; report OOD accuracy | Latency optimization to hit < 50 ms; p50/p95/p99 reporting |
| **6** | Clean-Traffic Safety | Build hard/realistic clean corpus | FPR tuning; ROC/AUC and operating-point selection | Dashboard: live FPR/TPR, attack-family distribution |
| **7** | Red-Team (Milestone: **adaptive break attempt**) | Implement white-box adaptive attacks vs. own detector | Analyze failures; attempt one hardening iteration | Fail-open/closed policy, alerting hook, deployment doc |
| **8** | Synthesis | Write arms-race narrative & results | Finalize generalization matrix & ablations | Package reference deployment; demo & final report |

**Key milestones:** end-to-end demo (Wk 4) · generalization matrix (Wk 5) · FPR-safe operating point (Wk 6) · adaptive break report (Wk 7) · final TDD + research write-up (Wk 8).

---

## 7. Risks, Ethics & Limitations

### 7.1 Technical Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Detector fails to generalize OOD | High (and *expected*) | Reframe as a research finding; report the matrix honestly |
| Adaptive attacks fully defeat the detector | Medium–High | Report the arms race and attacker-cost gains rather than claiming victory |
| Latency budget exceeded | Medium | Co-locate detector, batch, async logging, quantize the detector |
| High FPR blocks legitimate traffic | Medium | ROC-based threshold selection on a realistic clean corpus |

### 7.2 Ethics & Responsible Disclosure

- All attacks are generated against **the team's own models** in an isolated environment; no third-party production system is targeted.
- Jailbreak/injection corpora are drawn from **already-public** research datasets; no novel high-severity exploits are released.
- The deliverable is a **defensive** tool. Any attack code is scoped to evaluation of the team's own detector.

### 7.3 Honest Limitations

- An 8-week, 3-person effort delivers a **reference platform and a benchmark**, not a hardened commercial product.
- The single-node deployment demonstrates the architecture but is not multi-tenant or autoscaled.
- Certified/provable robustness is explicitly out of scope; SentinelML offers empirical, not guaranteed, defense.

---

## 8. References

*(Format for final submission per required citation style; representative foundational works.)*

1. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples* (FGSM). ICLR.
2. Madry, A., et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD). ICLR.
3. Athalye, A., Carlini, N., & Wagner, D. (2018). *Obfuscated Gradients Give a False Sense of Security.* ICML.
4. Carlini, N., & Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks* (C&W). IEEE S&P.
5. Tramèr, F., et al. (2020). *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS.
6. Xu, W., Evans, D., & Qi, Y. (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks.* NDSS.
7. Greshake, K., et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.*
8. OWASP. *Top 10 for Large Language Model Applications.*

---

*End of document — SentinelML TDD & Research Proposal v1.0.*
