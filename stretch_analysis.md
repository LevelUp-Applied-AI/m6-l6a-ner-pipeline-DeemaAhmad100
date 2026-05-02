# Stretch 6A-S1 — Custom NER Rules: Analysis

## What I Built

A custom spaCy `EntityRuler` with **20 patterns** across **4 custom labels**:

| Label | Examples | Count |
|-------|----------|-------|
| `CLIMATE_EVENT` | COP28, COP27, COP26, COP21, Climate Ambition Summit | 5 |
| `POLICY` | Paris Agreement, Kyoto Protocol, Glasgow Climate Pact, Kigali Amendment, Global Methane Pledge, Carbon Border Adjustment Mechanism | 6 |
| `REPORT` | IPCC AR6, IPCC AR5, Sixth Assessment Report, Emissions Gap Report | 4 |
| `THRESHOLD` | 1.5 degrees Celsius, 2 degrees Celsius, net zero, net-zero, carbon neutrality | 5 |

---

## Why These Patterns?

I started by running the base NER on 15 articles and manually checking
which climate terms it missed. The base model (`en_core_web_sm`) consistently
failed to recognize:

- **Paris Agreement** — missed in 3 articles ("Paris" tagged as GPE, "Agreement" dropped)
- **COP28** — missed in 2 articles (tagged as ORG or ignored)
- **1.5 degrees Celsius** — missed in 1 article (tagged as QUANTITY, losing policy significance)

These patterns are specific enough to avoid false positives:
- `"Paris"` alone → does NOT trigger POLICY (would match Paris the city)
- `"Paris Agreement"` → correctly triggers POLICY

---

## Before / After Comparison

| Label | Baseline | Ruler-before | Ruler-after |
|-------|----------|-------------|-------------|
| ORG | 184 | 181 | 184 |
| GPE | 165 | 164 | 165 |
| DATE | 256 | 256 | 256 |
| EVENT | 8 | 2 | 8 |
| LAW | 5 | 3 | 5 |
| QUANTITY | 92 | 90 | 92 |
| WORK_OF_ART | 6 | 5 | 6 |
| **CLIMATE_EVENT** | **0** | **+8** | **+5** |
| **POLICY** | **0** | **+10** | **0** |
| **REPORT** | **0** | **+2** | **0** |
| **THRESHOLD** | **0** | **+5** | **+3** |

**Key observation:** Ruler-before found all 25 custom entities but caused
the base NER to lose some standard entities (EVENT -6, ORG -3, LAW -2).
Ruler-after was safer — only added CLIMATE_EVENT and THRESHOLD, and left
all standard entities untouched.

---

## Precision / Recall / F1 on Gold Standard

*(Standard labels only: ORG, GPE, DATE, LAW, MONEY, PERSON, QUANTITY, LOC, EVENT, WORK_OF_ART)*

| Pipeline | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Baseline (no ruler) | 0.176 | 0.600 | 0.273 |
| Ruler before NER | 0.197 | 0.600 | 0.296 |
| Ruler after NER | 0.176 | 0.600 | 0.273 |

**Delta vs baseline:**
- Ruler before NER: P: +0.020  R: +0.000  F1: +0.024
- Ruler after NER:  P: +0.000  R: +0.000  F1: +0.000

**Interpretation:** Recall stayed at 0.600 across all pipelines because
the gold standard only contains standard labels — the custom labels are
not evaluated here. The small precision improvement in ruler-before
(+0.020) comes from the ruler correctly reclaiming spans that base NER
was tagging under wrong labels.

---

## Custom Label Examples (Qualitative)

**REPORT — Sixth Assessment Report** ✅ Correct
> *"The IPCC released its **Sixth Assessment Report** in March 2023"*
> Base NER split this: "IPCC"=ORG and "Sixth Assessment Report" dropped entirely.

**THRESHOLD — 1.5 degrees Celsius** ✅ Correct
> *"global temperatures could exceed **1.5 degrees Celsius** above pre-industrial levels"*
> Base NER tags this as QUANTITY (a measurement). The ruler identifies it
> as the Paris Agreement's key warming threshold — a policy concept.

**CLIMATE_EVENT — COP28** ✅ Correct
> *"At **COP28** in Dubai, over 190 nations agreed…"*
> Base NER tags COP28 as ORG. It is a climate summit (event), not an organization.

**POLICY — Paris Agreement** ✅ Correct
> *"contributions fall short of the **Paris Agreement** targets"*
> Base NER tags "Paris" as GPE and drops "Agreement". The ruler captures the full treaty name.

**POLICY — Carbon Border Adjustment Mechanism** ✅ Correct
> *"The European Union's **Carbon Border Adjustment Mechanism** entered its transitional phase"*
> Base NER ignores this multi-token policy name entirely.

**THRESHOLD — 2 degrees Celsius** ⚠️ False positive
> *"Water temperatures exceeded **2 degrees Celsius** above the seasonal average"* (Article 14)
> Here "2 degrees Celsius" refers to ocean temperature above a seasonal
> average — a scientific measurement, NOT the Paris Agreement threshold.
> The pattern fired on the correct string but the meaning is different.
> This is a context-blindness failure: string rules cannot distinguish intent.

---

## Analysis

The EntityRuler significantly improved recall for domain-specific climate
terminology that spaCy's `en_core_web_sm` was never trained to recognize.
The base model fragments multi-token phrases like "Paris Agreement" into
"Paris" (GPE) and a dropped token, losing the policy significance entirely.
The ruler-before configuration tagged 10 POLICY entities and 8 CLIMATE_EVENT
entities that the base NER missed completely — distinctions that matter for
a climate research application where "Paris" (a city) and "Paris Agreement"
(a treaty) carry fundamentally different analytical weight. The small but
real precision improvement on the gold standard (+0.020) reflects the ruler
correctly reclaiming spans that the base model was misclassifying.

Custom rules introduce noise in two predictable ways. First, string-based
patterns cannot distinguish context: "2 degrees Celsius" correctly tagged
the Paris Agreement threshold in Article 1 but incorrectly fired in
Article 14, where the same phrase described ocean temperature above a
seasonal average — a measurement, not a policy target. Second, the
ruler-before position caused the base NER to lose standard entities it
would otherwise have found (EVENT -6, ORG -3, LAW -2), because the ruler
consumed token spans that the NER model needed for contextual reasoning.
Ruler-after avoided this interference entirely but missed all POLICY
entities because the base NER had already consumed "Paris" as GPE before
the ruler could tag "Paris Agreement" as POLICY. The practical conclusion
is to use ruler-after for single-token terms like COP28 and THRESHOLD,
and ruler-before only for confirmed multi-token phrases where base NER
fragmentation is documented — accepting the tradeoff of minor standard-label
losses in exchange for correct multi-token policy identification.

---

## Pipeline Position Summary

| Position | Custom entities found | Standard entities lost | Best for |
|----------|-----------------------|----------------------|----------|
| **Before NER** | 25 | 13 | Multi-token phrases (Paris Agreement, Carbon Border Adjustment Mechanism) |
| **After NER** | 8 | 0 | Single-token terms (COP28, net zero) |