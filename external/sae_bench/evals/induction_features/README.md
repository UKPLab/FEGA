**Induction Feature Pipeline Note**

For SAE Geometry experiments, use the unified LSC/WC/TT/PrOntoQA workflow in
`external/sae_bench/evals/icl_features/README.md`. It enforces exact model-correct
example counts, prompt-family and support-slot balance, shared JSON fields, and
the 90%/90% pointer-like feature criterion. The code in this folder remains the
backward-compatible feature-discovery implementation used by that workflow.

We built a synthetic induction-style dataset plus an SAE feature analysis script to identify features that consistently activate when the model has to recover a hidden mapping from in-context examples.

**1. Dataset construction**

Code:
- `external/prontoqa/generate_induction_dataset.py`

What it does:
- Creates many 3-shot contexts.
- Each context contains prompt-local random source → target mappings like:
  `Every twimpee is a grimpant.`
- The final query repeats the source concept but hides the target, so the model must recover the mapping from the in-context support examples.
- We used `query_style=rule_completion` because it gives a cleaner induction-style repeated prefix before the answer.

Example generation command:
```bash
python3 external/prontoqa/generate_induction_dataset.py \
  --num-contexts 2000 \
  --queries-per-context 100 \
  --query-style rule_completion \
  --layout flat \
  --output data/prontoqa-1/prontoqa_induction_rule.json
```

Important fields in the JSON:
- `prompt`: full prompt shown to the model
- `answer`: target answer string
- `context_id`: groups queries that share the same 3-shot support set
- `support_example_index`: which support rule the query is reusing
- `induction_prefix`: repeated prefix immediately before the answer target

**2. Induction feature identification**

Code:
- `external/sae_bench/evals/induction_features/main.py`
- `external/sae_bench/evals/induction_features/analysis.py`

Helper used for custom SAE loading:
- `external/sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py`

High-level algorithm:
1. Load the dataset and tokenize each prompt.
2. For each prompt, run the base model and take the activation at the final prompt token, i.e. the token immediately before the first answer token.
3. Pass that activation through the SAE with `sae.encode(...)`.
4. Mark a feature as active if its post-encode activation is greater than the threshold.
   Default: `activation > 0.0`
5. Aggregate activity statistics per feature across all examples and across contexts.
6. Define a candidate induction feature as one that is:
   - active in at least `min_query_fraction_per_context` of the queries inside a context
   - and satisfies that in at least `min_context_fraction` of all analyzed contexts
7. Also compute `strict_common_feature_ids`, meaning features active on 100% of
   analyzed examples.

Current interpretation details:
- We usually run with `--require-model-correct`, which means only examples where the model gets the first answer token correct are included.
- For Gemma, do not use `--single-token-only` with this nonce dataset, because the answers are usually split into multiple subword tokens.

**3. How to run**

Run from the repo root.

Example run for the ReLU SAE we used:
```bash
USE_TORCH=1 USE_TF=0 python -m sae_bench.evals.induction_features.main \
  --dataset-path data/prontoqa-1/prontoqa_induction_rule.json \
  --output-dir data/induction_feature_outputs/gemma2b_relu_trainer5 \
  --model-name gemma-2-2b \
  --repo-id canrager/saebench_gemma-2-2b_width-2pow16_date-0107 \
  --sae-location gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_5 \
  --download-saes-dir data/downloaded_saes \
  --device cuda:0 \
  --batch-size 32 \
  --require-model-correct
```

Notes:
- `--model-name gemma-2-2b` requires Hugging Face access to `google/gemma-2-2b`
- we used the custom SAE repo:
  `canrager/saebench_gemma-2-2b_width-2pow16_date-0107`
- current SAE choice:
  `gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_5`

**4. Outputs**

Main output folder:
- `data/induction_feature_outputs/gemma2b_relu_trainer5/`

Files:
- `summary.json`
  Top-level summary. Best place to get:
  - candidate feature IDs
  - strict common feature IDs
  - total candidate count
  - candidate fraction of total SAE features
- `feature_metrics.csv`
  One row per SAE feature with all statistics.
- `candidate_features.csv`
  Filtered shortlist of candidate induction features.
- `sae_summary.csv`
  Per-SAE summary.
- `layer_summary.csv`
  Per-layer summary. This only becomes meaningful when we run multiple SAEs across layers.
- `plots/top_induction_features.png`
  Top-ranked candidate features by context commonality.
- `plots/feature_prevalence_overview.png`
  Distribution of feature commonality and activation strength.
- `plots/candidate_features_by_layer.png`
  Layer distribution of candidate counts.

**5. Most important columns for downstream analysis**

From `feature_metrics.csv` or `candidate_features.csv`:
- `feature_id`: SAE feature index
- `consistent_context_prevalence`: main score for “common across contexts”
- `num_consistent_contexts`: raw count behind that score
- `example_prevalence`: fraction of analyzed examples where the feature is active
- `mean_activation_when_active`: average magnitude when active
- `is_candidate_feature`: passed threshold rule
- `is_strict_common_feature`: active under the rule in all analyzed contexts
- `slot_0_example_prevalence`, `slot_1_example_prevalence`, `slot_2_example_prevalence`:
  useful for checking whether a feature is general across support slots or tied to one slot only


**6. Interpretation**

The strongest induction-feature candidates are features with:
- high `consistent_context_prevalence`
- ideally `is_strict_common_feature = True`
- reasonably high `mean_activation_when_active`
- and balanced slot prevalences rather than firing only for one support-example slot
