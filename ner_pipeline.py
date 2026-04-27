"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import unicodedata

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    return pd.read_csv(filepath)

def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df['text'].apply(lambda x: len(x.split()))
    return {
       'shape': tuple(df.shape),

        'lang_counts': df['language'].value_counts().to_dict(),
        'category_counts': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': word_counts.mean(),
            'min': word_counts.min(),
            'max': word_counts.max()
        }
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
   
    text = unicodedata.normalize('NFC', text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        tokens.append(token.lemma_.lower())
    return tokens

def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """




    rows = []
    english_df = df[df['language'] == 'en']
    for _, row in english_df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            rows.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    return pd.DataFrame(rows)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    rows = []
    english_df = df[df['language'] == 'en']
    for _, row in english_df.iterrows():
        results = ner_pipeline(row['text'])
        
       
        merged = []
        for item in results:
            if item['word'].startswith('##') and merged:
                merged[-1]['word'] += item['word'][2:]
            else:
                merged.append(dict(item))
        
        for item in merged:
            label = item['entity'].replace('B-', '').replace('I-', '')
            rows.append({
                'text_id': row['id'],
                'entity_text': item['word'],
                'entity_label': label,
                'start_char': item['start'],
                'end_char': item['end']
            })
    return pd.DataFrame(rows)

def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text']))
    return {
        'spacy_counts': spacy_df['entity_label'].value_counts().to_dict(),
        'hf_counts': hf_df['entity_label'].value_counts().to_dict(),
        'total_spacy': len(spacy_df),
        'total_hf': len(hf_df),
        'both': spacy_set & hf_set,
        'spacy_only': spacy_set - hf_set,
        'hf_only': hf_set - spacy_set
    }

def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
   
    if predicted_df is None or gold_df is None:
        return None
    if len(predicted_df) == 0 or len(gold_df) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    pred_set = set(zip(
        predicted_df['text_id'].astype(str),
        predicted_df['entity_text'].astype(str),
        predicted_df['entity_label'].astype(str)
    ))
    gold_set = set(zip(
        gold_df['text_id'].astype(str),
        gold_df['entity_text'].astype(str),
        gold_df['entity_label'].astype(str)
    ))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1}



def analyze_by_category(df, spacy_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    merged = spacy_df.merge(
        df[['id', 'category']], 
        left_on='text_id', 
        right_on='id'
    )

    pivot = merged.groupby(['category', 'entity_label']).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues')
    plt.title('Entity Types by Category')
    plt.tight_layout()
    plt.savefig('category_entity_heatmap.png')
    print("\nTier 1 — Entity counts by category:")
    print(pivot)
    return pivot

    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def entity_aggregation_pipeline(df, spacy_df):
    import matplotlib.pyplot as plt
    from itertools import combinations
    from collections import defaultdict

    # ① Entity Normalization
    normalization_map = {
        "United Nations": "UN",
        "U.N.": "UN",
        "the UN": "UN",
        "European Union": "EU",
        "E.U.": "EU",
        "Hugging Face": "HF",
        "United States": "US",
        "U.S.": "US",
        "United Arab Emirates": "UAE",
        "U.A.E.": "UAE",
    }

    spacy_df = spacy_df.copy()
    spacy_df['entity_norm'] = spacy_df['entity_text'].apply(
        lambda x: normalization_map.get(x, x)
    )

    print("\nTier 2 — Sample normalizations:")
    for orig, norm in normalization_map.items():
        count = (spacy_df['entity_text'] == orig).sum()
        if count > 0:
            print(f"  '{orig}' → '{norm}' ({count} times)")

    # ② Entity Co-occurrence
    co_occur = defaultdict(int)
    for text_id, group in spacy_df.groupby('text_id'):
        entities = group['entity_norm'].unique().tolist()
        for e1, e2 in combinations(sorted(entities), 2):
            co_occur[(e1, e2)] += 1

    # Top 20 co-occurrences
    top_pairs = sorted(co_occur.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 10 entity co-occurrences:")
    for (e1, e2), count in top_pairs[:10]:
        print(f"  {e1} + {e2}: {count}")

    # ③ TF-IDF Entity Importance
    entity_doc_counts = spacy_df.groupby('entity_norm')['text_id'].nunique()
    total_docs = spacy_df['text_id'].nunique()
    import numpy as np
    idf = np.log(total_docs / (entity_doc_counts + 1))
    tf = spacy_df['entity_norm'].value_counts()
    tfidf = (tf * idf).dropna().sort_values(ascending=False)

    print("\nTop 10 distinctive entities (TF-IDF):")
    print(tfidf.head(10))

    # ④ Network Visualization
    nodes = set()
    for (e1, e2), _ in top_pairs:
        nodes.add(e1)
        nodes.add(e2)
    node_list = list(nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    import math
    n = len(node_list)
    angles = [2 * math.pi * i / n for i in range(n)]
    pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(node_list, angles)}

    max_count = top_pairs[0][1]
    for (e1, e2), count in top_pairs:
        x = [pos[e1][0], pos[e2][0]]
        y = [pos[e1][1], pos[e2][1]]
        ax.plot(x, y, 'b-', alpha=0.3, linewidth=count/max_count*3)

    for node, (x, y) in pos.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.annotate(node, (x, y), fontsize=7, ha='center',
                   xytext=(x*1.15, y*1.15))

    ax.set_title('Top 20 Entity Co-occurrences Network')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('entity_cooccurrence_network.png')
    print("\nNetwork saved: entity_cooccurrence_network.png")

    return {'tfidf': tfidf, 'co_occurrences': top_pairs}



def custom_ner_evaluator(predicted_df, gold_df):
    """Tier 3: Comprehensive NER evaluation with 3 matching strategies."""
    from collections import defaultdict

    pred_list = list(zip(predicted_df['text_id'],
                         predicted_df['entity_text'],
                         predicted_df['entity_label']))
    gold_list = list(zip(gold_df['text_id'],
                         gold_df['entity_text'],
                         gold_df['entity_label']))

    pred_set = set(pred_list)
    gold_set = set(gold_list)

    # ① Exact Match — نص + label لازم يتطابقوا
    exact_tp = len(pred_set & gold_set)
    exact_fp = len(pred_set - gold_set)
    exact_fn = len(gold_set - pred_set)
    exact_p = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0
    exact_r = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0
    exact_f1 = (2 * exact_p * exact_r / (exact_p + exact_r)
                if (exact_p + exact_r) > 0 else 0)

    # ② Type-Agnostic Match — نفس النص بغض النظر عن الـ label
    pred_spans = set(zip(predicted_df['text_id'], predicted_df['entity_text']))
    gold_spans = set(zip(gold_df['text_id'], gold_df['entity_text']))
    agnostic_tp = len(pred_spans & gold_spans)
    agnostic_fp = len(pred_spans - gold_spans)
    agnostic_fn = len(gold_spans - pred_spans)
    agnostic_p = agnostic_tp / (agnostic_tp + agnostic_fp) if (agnostic_tp + agnostic_fp) > 0 else 0
    agnostic_r = agnostic_tp / (agnostic_tp + agnostic_fn) if (agnostic_tp + agnostic_fn) > 0 else 0
    agnostic_f1 = (2 * agnostic_p * agnostic_r / (agnostic_p + agnostic_r)
                   if (agnostic_p + agnostic_r) > 0 else 0)

    # ③ Partial Match — جزء من النص يكفي
    partial_tp = 0
    for p_id, p_text, p_label in pred_list:
        for g_id, g_text, g_label in gold_list:
            if p_id == g_id and (p_text in g_text or g_text in p_text):
                partial_tp += 1
                break
    partial_p = partial_tp / len(pred_list) if pred_list else 0
    partial_r = partial_tp / len(gold_list) if gold_list else 0
    partial_f1 = (2 * partial_p * partial_r / (partial_p + partial_r)
                  if (partial_p + partial_r) > 0 else 0)

    # ④ Error Analysis
    errors = defaultdict(int)
    for g_id, g_text, g_label in gold_list:
        matched_span = any(p_id == g_id and p_text == g_text
                          for p_id, p_text, _ in pred_list)
        matched_exact = (g_id, g_text, g_label) in pred_set

        if matched_exact:
            continue
        elif matched_span:
            errors['type_error'] += 1
        else:
            partial = any(p_id == g_id and (p_text in g_text or g_text in p_text)
                         for p_id, p_text, _ in pred_list)
            if partial:
                errors['boundary_error'] += 1
            else:
                errors['missing_entity'] += 1

    for p_id, p_text, p_label in pred_list:
        if (p_id, p_text, p_label) not in gold_set:
            errors['spurious_entity'] += 1

    print("\nTier 3 — Custom NER Evaluator:")
    print(f"\n  Exact Match:        P={exact_p:.3f} R={exact_r:.3f} F1={exact_f1:.3f}")
    print(f"  Type-Agnostic:      P={agnostic_p:.3f} R={agnostic_r:.3f} F1={agnostic_f1:.3f}")
    print(f"  Partial Match:      P={partial_p:.3f} R={partial_r:.3f} F1={partial_f1:.3f}")
    print(f"\n  Error Analysis:")
    for error_type, count in sorted(errors.items()):
        print(f"    {error_type}: {count}")

    return {
        'exact': {'precision': exact_p, 'recall': exact_r, 'f1': exact_f1},
        'type_agnostic': {'precision': agnostic_p, 'recall': agnostic_r, 'f1': agnostic_f1},
        'partial': {'precision': partial_p, 'recall': partial_r, 'f1': partial_f1},
        'errors': dict(errors)
    }


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        gold = pd.read_csv("data/gold_entities.csv")

        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")

        if hf_entities is not None:          
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation: {hf_metrics}")

                   # Tier 1
        if spacy_entities is not None:
            analyze_by_category(df, spacy_entities)

            # Tier 2
        if spacy_entities is not None:
            entity_aggregation_pipeline(df, spacy_entities)

              # Tier 3
        if spacy_entities is not None:
            custom_ner_evaluator(spacy_entities, gold)