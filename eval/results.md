# Evaluation: our method vs BERTopic (70 held-out gold_test reviews)

Gold labels are LLM-generated, not hand-written (human spot-check: 20/20); see `gold_labeling_notes.md`.

| Method | Recall (all gold items) | Recall on multi-complaint reviews | Precision |
|---|---|---|---|
| Ours, phrase level | 72.4% | 73.3% | 90.1% strict / 97.5% with judge |
| Ours, theme level | 43.8% | 44.0% | n/a |
| BERTopic (one topic per review) | 15.2% | 17.3% | n/a |

Gold items: 105; multi-complaint reviews: 29. BERTopic: 7 topics, 49 outliers reassigned. Matching: cosine >= 0.8 match, < 0.6 no match, in between judged by gpt-oss-120b (a different model family from the extractor).