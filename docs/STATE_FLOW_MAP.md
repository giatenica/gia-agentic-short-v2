# Workflow State Flow Map

Current date: 2026-01-04

This document captures the practical state contract across nodes: what each node reads from `WorkflowState` and what it writes back. The goal is to prevent silent context loss (for example, writer prompts becoming ungrounded due to missing keys).

## Canonical keys

These are the keys that downstream nodes should prefer:

- Research question: `refined_query` (fallback: `original_query`)
- Literature: `search_results` and `literature_synthesis`
- Gap: `gap_analysis`, `identified_gaps`, `contribution_statement`
- Plan: `research_plan`
- Analysis: `analysis`
- Writing: `writer_output`, `completed_sections`, `reference_list`, `style_violations`
- Review: `review_critique`, `review_decision`, `revision_request`, `reviewer_output`, `human_feedback`

Legacy keys still appear in a few places for backward compatibility:

- `data_analyst_output`, `conceptual_synthesis_output`
- `research_intake` (writer test back-compat)

## Node contracts

### INTAKE

Reads:
- `form_data`

Writes:
- `original_query`, `project_title`, `target_journal`, `paper_type`, `research_type`
- `user_hypothesis`, `proposed_methodology`, `seed_literature`, `expected_contribution`
- `constraints`, `deadline`, `uploaded_data`, `data_context`, `key_variables`
- `status`, `errors`, `messages`, `checkpoints`, `updated_at`

File: src/nodes/intake.py

### DATA_EXPLORER

Reads:
- `uploaded_data`, `key_variables`, `original_query`

Writes:
- `data_exploration_results`, `data_exploration_summary`, `loaded_datasets`
- also emits non-canonical helper keys: `all_exploration_results`, `variable_mappings`
- `status`, `errors`, `messages`, `checkpoints`, `updated_at`

File: src/nodes/data_explorer.py

### LITERATURE_REVIEWER

Reads:
- `original_query`, `key_variables`

Writes:
- `search_results`, `seed_literature`, `methodology_precedents`
- `status`, `messages`, `checkpoints`, `updated_at`

File: src/nodes/literature_reviewer.py

### LITERATURE_SYNTHESIZER

Reads:
- `search_results`, `original_query`

Writes:
- `literature_synthesis` (dict shape aligned with `LiteratureSynthesis` expectations)
- `literature_themes`, `identified_gaps`, `contribution_statement`, `refined_query`
- `status`, `messages`, `checkpoints`, `updated_at`

File: src/nodes/literature_synthesizer.py

### GAP_IDENTIFIER

Reads:
- `original_query`
- prefers `literature_synthesis`; if missing, falls back to `search_results`
- optional: `data_exploration_results`, `expected_contribution`

Writes:
- `gap_analysis` (dict), `identified_gaps` (list[str])
- `refined_query`, `contribution_statement`
- `status`, `messages`

File: src/nodes/gap_identifier.py

### PLANNER

Reads:
- research question: `refined_query` / `refined_research_question` / `original_query`
- `gap_analysis`, `literature_synthesis`, `data_exploration_results`
- optional: `research_type`, `paper_type`, `contribution_statement`, `deadline`, `key_variables`

Writes:
- `research_plan`
- `status`, `messages` (HITL interrupt occurs inside the node)

File: src/nodes/planner.py

### DATA_ACQUISITION

Reads:
- `research_plan`, `loaded_datasets`, `research_type`

Writes:
- `data_acquisition_plan`, `acquired_datasets`, `acquisition_failures`, `generated_code_snippets`
- `loaded_datasets` (augmented), `messages`

File: src/nodes/data_acquisition.py

### DATA_ANALYST

Reads:
- `research_plan`, `loaded_datasets`, `uploaded_data`
- `data_exploration_results`
- `gap_analysis`, `refined_query` / `refined_research_question` / `original_query`

Writes:
- `analysis` (canonical)
- legacy routing key: `data_analyst_output` (copy of `analysis` as dict)
- `tables`, `figures`, `status`, `messages`

File: src/nodes/data_analyst.py

### CONCEPTUAL_SYNTHESIZER

Reads:
- `literature_synthesis`, `gap_analysis`
- `refined_query` / `refined_research_question` / `original_query`

Writes:
- `analysis` (canonical)
- `status`, `messages`

File: src/nodes/conceptual_synthesizer.py

### WRITER

Reads:
- paper metadata: `project_title`, `target_journal`, `paper_type`, `research_type`
- question: `refined_query` / `original_query` (legacy fallback: `research_intake`)
- `research_plan`, `literature_synthesis`, `gap_analysis`, `identified_gaps`
- analysis: `analysis` (legacy fallbacks: `data_analyst_output`, `conceptual_synthesis_output`)
- `tables`, `figures`, `data_exploration_summary`
- revision loop: `review_decision`, `revision_request`, `reviewer_output`, `human_feedback`
- `sections_to_write`, `completed_sections`, `writer_output`

Writes:
- `writer_output`, `completed_sections`, `reference_list`, `style_violations`

File: src/nodes/writer.py

### REVIEWER

Reads:
- `writer_output`
- `original_query`, `identified_gaps`, `research_plan`, `analysis`, `target_journal`
- `revision_count`, `max_revisions`

Writes:
- `review_critique`, `review_decision`, `revision_request`, `reviewer_output`
- `revision_count`, `human_approved`, `human_feedback`
- `status`, `errors`, `messages`

File: src/nodes/reviewer.py

### FALLBACK

Reads:
- `errors`, plus whatever is available (`original_query`/`refined_query`, `literature_synthesis`/`search_results`, `research_plan`, `analysis`, `writer_output`)

Writes:
- `fallback_report`, `final_paper`, `_fallback_activated`, `status`

File: src/nodes/fallback.py

## Known non-canonical keys to watch

- `all_exploration_results`, `variable_mappings` are produced by `data_explorer_node` and are not required by downstream nodes.
- `data_analyst_output` exists for routing backward compatibility; downstream should prefer `analysis`.
