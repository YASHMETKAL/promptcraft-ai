PromptCraft AI — Custom Prompt Optimization Dataset
===================================================

Files
-----
- promptcraft_custom_dataset.csv: convenient for reviewing/editing in Excel or Google Sheets.
- promptcraft_custom_dataset.jsonl: one JSON record per line; convenient for ML pipelines.

Current size
------------
82 examples.

Purpose
-------
Each record pairs a realistic short user request with a clearer, more specific target prompt.
The examples cover education, coding, research, writing, presentations, image generation,
data analysis, career preparation, security, productivity, creative work and health information.

Important provenance note
-------------------------
These are original AI-assisted draft examples written to be varied and natural in style.
They are NOT independently human-authored or human-verified records. The field
'review_status' explicitly marks each record as requiring human review. Before using this
dataset for academic claims, review and edit every record, correct any weak examples,
and document the review process honestly.

Suggested next steps
--------------------
1. Read every row and revise examples that do not sound natural to you.
2. Check that each target prompt preserves the input's intent and adds only useful detail.
3. Add examples from your own domain and writing style.
4. Remove duplicate or near-duplicate examples.
5. Split by prompt/topic family into train, validation and test sets before training.
   Suggested starting split: 70% / 15% / 15%. Keep related examples in one split.
6. Do not use the held-out test set to train, tune prompts or select checkpoints.
7. Treat 'quality_label' as an initial draft label, not a verified ground-truth score.
   Replace it with human ratings after using a defined rubric.

Recommended fine-tuning fields
------------------------------
For supervised fine-tuning, use:
- input_prompt as the user message
- target_prompt as the assistant response

You can include category, intended_audience and requirements in the input template if you
want the model to condition on them. Avoid training directly on id, quality_label or
review_status unless your chosen task explicitly needs those fields.

Suggested review rubric (1–5 each)
----------------------------------
- Intent preserved
- Clarity
- Useful specificity
- Constraints included
- Naturalness

Do not claim that this dataset is human-written unless a human actually wrote the records.
