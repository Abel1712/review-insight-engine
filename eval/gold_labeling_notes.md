# Gold set: provenance and labeling rules

**Who labeled it:** Claude (Anthropic, `claude-opus-5-5`), on 2026-10-06, at Abel's explicit request.
**Not hand-labeled.** Any README, brief or resume text must say "labeled by Claude (an AI)", never "hand-labeled".

**Why this is still a meaningful check (and its limits):**
- The labeler is a different model family from every model in the pipeline (OpenAI gpt-oss, Alibaba Qwen, Google Gemini), so this is a cross-family comparison, not a model grading itself.
- Labels were written before any extraction ran on real reviews, so they can't copy the pipeline's output.
- Limit: LLMs share habits (phrasing, what they split or skip), so agreement may be higher than it would be with a human labeler. Recall numbers should be read with that in mind.
- Recommended: Abel spot-checks 20 labels; the human-AI agreement rate goes in `metrics.json`.

**Rules applied to all 100 reviews:**
1. Label every specific complaint and every specific praise, one issue per phrase, in plain English (any source language: Hindi, Hinglish, Telugu, Malayalam).
2. Skip pure overall sentiment with no target ("good app", "worst service", "very 3rd class app"). A review with only that is `none`.
3. An opinion needs a subject to count: the ride, the driver/captain, price/fares, earnings, support, a specific app behaviour.
4. Sarcasm is labeled by its meaning ("₹50 order, ₹29 commission. Super" = high commission complaint).
5. Suggestions count as items ("request to launch in pune").

Split: `gold_dev.csv` (30, 15 per app) for tuning and thresholds; `gold_test.csv` (70) untouched until Step 7.
