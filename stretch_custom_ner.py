"""
Stretch 6A-S1 — Custom NER Rules with EntityRuler
 
Extends spaCy's base NER with domain-specific climate terminology.
Tests EntityRuler in two positions (before/after NER) and evaluates
precision, recall, and F1 on the gold standard.
 
Run: python stretch_custom_ner.py
"""
 
import pandas as pd
import spacy
from spacy.scorer import Scorer
from spacy.training import Example
 
 
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
 
def load_data():
    """Load climate articles and gold standard entities."""
    articles = pd.read_csv("data/climate_articles.csv")
    gold     = pd.read_csv("data/gold_entities.csv")
    print(f"Loaded {len(articles)} articles")
    print(f"Loaded {len(gold)} gold entities")
    return articles, gold
 
 
# ─────────────────────────────────────────────
# 2. CUSTOM PATTERNS (10+ entries, 3+ labels)
# ─────────────────────────────────────────────
#
# WHY THESE LABELS?
#   CLIMATE_EVENT  → named climate conferences/summits
#   POLICY         → treaties, agreements, frameworks
#   REPORT         → named scientific assessments
#   THRESHOLD      → quantitative climate targets
#
# WHY SPECIFIC PATTERNS?
#   "Paris" alone would match Paris the city → wrong
#   "Paris Agreement" is specific → correct
#
# Each line below = 1 pattern entry
 
CUSTOM_PATTERNS = [
    # ── CLIMATE_EVENT (5 entries, 5 distinct concepts) ──
    {"label": "CLIMATE_EVENT", "pattern": "COP28"},           # 1
    {"label": "CLIMATE_EVENT", "pattern": "COP27"},           # 2
    {"label": "CLIMATE_EVENT", "pattern": "COP26"},           # 3
    {"label": "CLIMATE_EVENT", "pattern": "COP21"},           # 4
    {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},  # 5
 
    # ── POLICY (6 entries, 6 distinct concepts) ──
    {"label": "POLICY", "pattern": "Paris Agreement"},        # 6
    {"label": "POLICY", "pattern": "Kyoto Protocol"},         # 7
    {"label": "POLICY", "pattern": "Glasgow Climate Pact"},   # 8
    {"label": "POLICY", "pattern": "Kigali Amendment"},       # 9
    {"label": "POLICY", "pattern": "Global Methane Pledge"},  # 10
    {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},  # 11
 
    # ── REPORT (4 entries, 4 distinct concepts) ──
    {"label": "REPORT", "pattern": "IPCC AR6"},               # 12
    {"label": "REPORT", "pattern": "IPCC AR5"},               # 13
    {"label": "REPORT", "pattern": "Sixth Assessment Report"},# 14
    {"label": "REPORT", "pattern": "Emissions Gap Report"},   # 15
 
    # ── THRESHOLD (5 entries, 4 distinct concepts) ──
    {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"}, # 16
    {"label": "THRESHOLD", "pattern": "2 degrees Celsius"},   # 17
    {"label": "THRESHOLD", "pattern": "net zero"},            # 18
    {"label": "THRESHOLD", "pattern": "net-zero"},            # 19
    {"label": "THRESHOLD", "pattern": "carbon neutrality"},   # 20
]
 
print(f"\nTotal patterns: {len(CUSTOM_PATTERNS)}")
print(f"Labels: {sorted({p['label'] for p in CUSTOM_PATTERNS})}")
 
 
# ─────────────────────────────────────────────
# 3. BUILD PIPELINES
# ─────────────────────────────────────────────
 
def make_pipeline_before():
    """
    EntityRuler BEFORE the NER model.
 
    Timeline: text → ruler → NER → result
    Effect:   ruler marks entities first, NER cannot override them.
    Best for: multi-token phrases NER tends to split (e.g. "Paris Agreement"
              gets split into "Paris"=GPE + "Agreement"=ignored).
    Risk:     ruler may block NER from seeing context it needs.
    """
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner", name="ruler_before")
    ruler.add_patterns(CUSTOM_PATTERNS)
    return nlp
 
 
def make_pipeline_after():
    """
    EntityRuler AFTER the NER model.
 
    Timeline: text → NER → ruler → result
    Effect:   NER runs first; ruler only fills gaps NER missed.
    Best for: terms that NER partially catches under wrong labels.
    Risk:     cannot reclaim spans NER already consumed.
    """
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", after="ner", name="ruler_after")
    ruler.add_patterns(CUSTOM_PATTERNS)
    return nlp
 
 
# ─────────────────────────────────────────────
# 4. SPOT-CHECK: what does base NER miss?
# ─────────────────────────────────────────────
 
def spot_check(articles_df, nlp_base):
    """
    Run base NER on 15 English articles and show missed climate terms.
    This is how we decided WHICH patterns to add.
    """
    TARGET_TERMS = [
        "COP28", "COP27", "COP26", "Paris Agreement",
        "Kyoto Protocol", "IPCC AR6", "net zero", "net-zero",
        "1.5 degrees Celsius", "Glasgow Climate Pact",
        "Kigali Amendment", "Emissions Gap Report",
        "Carbon Border Adjustment Mechanism", "Global Methane Pledge",
        "Sixth Assessment Report",
    ]
 
    english = articles_df[articles_df["language"] == "en"].head(15)
    missed  = {term: 0 for term in TARGET_TERMS}
 
    for _, row in english.iterrows():
        doc = nlp_base(row["text"])
        found_labels = {ent.text: ent.label_ for ent in doc.ents}
        for term in TARGET_TERMS:
            if term in row["text"] and term not in found_labels:
                missed[term] += 1
 
    print("\n── Spot-check: terms missed by base NER ──────────────────")
    for term, count in sorted(missed.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  '{term}' missed in {count} article(s)")
 
 
# ─────────────────────────────────────────────
# 5. BEFORE/AFTER COMPARISON
# ─────────────────────────────────────────────
 
def count_labels(nlp, articles_df):
    """Count entity labels across all English articles."""
    counts: dict[str, int] = {}
    english = articles_df[articles_df["language"] == "en"]
    for doc in nlp.pipe(english["text"].tolist(), batch_size=50):
        for ent in doc.ents:
            counts[ent.label_] = counts.get(ent.label_, 0) + 1
    return counts
 
 
def print_comparison(base_counts, before_counts, after_counts):
    """Print a three-column comparison table."""
    all_labels = sorted(
        set(base_counts) | set(before_counts) | set(after_counts)
    )
    print(f"\n── Entity counts by label ────────────────────────────────")
    print(f"{'Label':<28} {'Baseline':>10} {'Before':>8} {'After':>7}")
    print("-" * 57)
    for lbl in all_labels:
        b  = base_counts.get(lbl, 0)
        bf = before_counts.get(lbl, 0)
        af = after_counts.get(lbl, 0)
        print(f"  {lbl:<26} {b:>10,} {bf:>8,} {af:>7,}")
 
    print(f"\n── Deltas vs baseline (ruler-before) ─────────────────────")
    for lbl in all_labels:
        d = before_counts.get(lbl, 0) - base_counts.get(lbl, 0)
        if d != 0:
            sign = "+" if d > 0 else ""
            print(f"  {lbl:<26} {sign}{d}")
 
    print(f"\n── Deltas vs baseline (ruler-after) ──────────────────────")
    for lbl in all_labels:
        d = after_counts.get(lbl, 0) - base_counts.get(lbl, 0)
        if d != 0:
            sign = "+" if d > 0 else ""
            print(f"  {lbl:<26} {sign}{d}")
 
 
# ─────────────────────────────────────────────
# 6. EVALUATION ON GOLD STANDARD
# ─────────────────────────────────────────────
 
# Labels that exist in both gold standard AND spaCy's base model.
# We evaluate ONLY on these to avoid artificially depressing precision
# with custom labels (CLIMATE_EVENT, POLICY, etc.) that the gold
# standard never uses.
STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART",
}
 
 
def build_gold_examples(gold_df, articles_df, nlp):
    """
    Convert gold_entities.csv into spaCy Example objects.
 
    gold_entities.csv columns: text_id, entity_text, entity_label,
                                start_char, end_char
    We filter to STANDARD_LABELS only.
    """
    examples = []
    # Group gold annotations by article
    grouped = gold_df[
        gold_df["entity_label"].isin(STANDARD_LABELS)
    ].groupby("text_id")
 
    for text_id, group in grouped:
        # Find the article text
        row = articles_df[articles_df["id"] == text_id]
        if row.empty:
            continue
        text = row.iloc[0]["text"]
 
        # Build predicted doc
        doc_pred = nlp(text)
 
        # Build reference doc with gold spans
        doc_ref = nlp.make_doc(text)
        spans   = []
        for _, ent_row in group.iterrows():
            span = doc_ref.char_span(
                int(ent_row["start_char"]),
                int(ent_row["end_char"]),
                label=ent_row["entity_label"],
            )
            if span is not None:
                spans.append(span)
 
        # Filter predicted entities to STANDARD_LABELS only
        doc_pred.ents = [
            e for e in doc_pred.ents
            if e.label_ in STANDARD_LABELS
        ]
 
        doc_ref.ents = spans
        examples.append(Example(doc_pred, doc_ref))
 
    return examples
 
 
def evaluate(nlp, gold_df, articles_df, label="pipeline"):
    """Score a pipeline on the gold standard (standard labels only)."""
    examples = build_gold_examples(gold_df, articles_df, nlp)
    scorer   = Scorer()
    scores   = scorer.score(examples)
    p  = scores.get("ents_p", 0.0)
    r  = scores.get("ents_r", 0.0)
    f1 = scores.get("ents_f", 0.0)
    print(f"  {label:<22}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    return {"precision": p, "recall": r, "f1": f1}
 
 
# ─────────────────────────────────────────────
# 7. QUALITATIVE CHECK FOR CUSTOM LABELS
# ─────────────────────────────────────────────
 
def show_custom_label_examples(nlp, articles_df, n_articles=20):
    """
    Show example sentences where custom rules fired.
    Used for qualitative evaluation of custom labels.
    """
    custom_labels = {"CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD"}
    english = articles_df[articles_df["language"] == "en"].head(n_articles)
 
    print(f"\n── Custom label examples ─────────────────────────────────")
    found_any = False
    for _, row in english.iterrows():
        doc = nlp(row["text"])
        custom_ents = [e for e in doc.ents if e.label_ in custom_labels]
        if custom_ents:
            found_any = True
            print(f"\n  Article {row['id']} [{row['category']}]:")
            for ent in custom_ents:
                # Show the entity with 30 chars of surrounding context
                start  = max(0, ent.start_char - 30)
                end    = min(len(row["text"]), ent.end_char + 30)
                ctx    = row["text"][start:end].replace("\n", " ")
                print(f"    [{ent.label_}] '{ent.text}'")
                print(f"    context: '…{ctx}…'")
 
    if not found_any:
        print("  No custom label entities found in first 20 articles.")
 
 
# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
 
    # ── Load data ────────────────────────────────────────────────────
    articles_df, gold_df = load_data()
 
    # ── Build three separate pipelines ───────────────────────────────
    print("\nLoading pipelines…")
    nlp_base   = spacy.load("en_core_web_sm")   # no ruler
    nlp_before = make_pipeline_before()          # ruler before NER
    nlp_after  = make_pipeline_after()           # ruler after NER
 
    # ── Spot-check: what does base NER miss? ─────────────────────────
    spot_check(articles_df, nlp_base)
 
    # ── Before/after comparison ───────────────────────────────────────
    print("\nCounting entities across all English articles…")
    base_counts   = count_labels(nlp_base,   articles_df)
    before_counts = count_labels(nlp_before, articles_df)
    after_counts  = count_labels(nlp_after,  articles_df)
    print_comparison(base_counts, before_counts, after_counts)
 
    # ── Gold standard evaluation (standard labels only) ───────────────
    print(f"\n── Gold standard evaluation (standard labels only) ───────")
    base_scores   = evaluate(nlp_base,   gold_df, articles_df, "Baseline (no ruler)")
    before_scores = evaluate(nlp_before, gold_df, articles_df, "Ruler before NER")
    after_scores  = evaluate(nlp_after,  gold_df, articles_df, "Ruler after NER")
 
    print(f"\n── Deltas vs baseline ────────────────────────────────────")
    for name, sc in [("Ruler before NER", before_scores),
                     ("Ruler after NER",  after_scores)]:
        dp = sc["precision"] - base_scores["precision"]
        dr = sc["recall"]    - base_scores["recall"]
        df = sc["f1"]        - base_scores["f1"]
        sp = "+" if dp >= 0 else ""
        sr = "+" if dr >= 0 else ""
        sf = "+" if df >= 0 else ""
        print(f"  {name:<22}  "
              f"P: {sp}{dp:.3f}  R: {sr}{dr:.3f}  F1: {sf}{df:.3f}")
 
    # ── Qualitative: custom label examples ────────────────────────────
    show_custom_label_examples(nlp_before, articles_df, n_articles=30)
 
    print("\nDone! Run stretch_analysis.md for the written analysis.")