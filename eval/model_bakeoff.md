# Model bake-off (gold_dev, 30 reviews)

Gold labels: 41 items, LLM-generated (see gold_labeling_notes.md). Recall = share of gold items matched by an extracted phrase from the same review (cosine >= t).

| model               | provider   | family   |   recall@0.6 |   recall@0.7 |   recall@0.8 |   first_pass_valid |   failed |   evidence_found |   phrases |   unsure_reviews |   seconds |
|:--------------------|:-----------|:---------|-------------:|-------------:|-------------:|-------------------:|---------:|-----------------:|----------:|-----------------:|----------:|
| qwen/qwen3.8-27b    | groq       | qwen     |        0.854 |        0.756 |        0.561 |              1     |        0 |            1     |        35 |                4 |       0   |
| openai/gpt-oss-120b | groq       | openai   |        0.732 |        0.634 |        0.512 |              1     |        0 |            1     |        35 |                2 |       0.1 |
| openai/gpt-oss-20b  | groq       | openai   |        0.61  |        0.488 |        0.341 |              0.967 |        0 |            0.968 |        31 |                0 |       0.1 |

**Main:** qwen/qwen3.8-27b (best Groq model by recall@0.7, then evidence accuracy)  
**Secondary:** openai/gpt-oss-120b (best model from a different family)

**Could not be scored:** gemini-3.8-flash (NoModelAvailable)