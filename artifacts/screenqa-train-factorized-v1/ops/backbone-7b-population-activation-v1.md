# ScreenQA Qwen-7B diagnostic population activation v1

Status: activated on 2026-08-31 after the outcome-blind 512-source selection
completed and before any Qwen-7B rollout or answer likelihood was computed for
the selected population.

## Frozen inputs and outputs

- Protocol SHA-256:
  `1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474`.
- Parent ScreenQA ranker-development manifest SHA-256:
  `a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec`.
- Selected 512-source manifest:
  `artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/backbone-7b-source512-manifest-v1.jsonl`.
- Selected manifest SHA-256:
  `4af43ac80a1666c174774d1c33383adca625e1ef4fc535ffb74e627f149290d0`.
- Selection report SHA-256:
  `47a75848ed99ee21c1d0a0440b05ef3dd8575d7bd5cf3e9ef0679d43e2213504`.
- Selector implementation revision:
  `7b8f5e803224e10f426c2de5257b40af23058c05`.

## Independent structural audit

- Exactly 512 rows, 512 unique source IDs, 512 unique state IDs, and 512
  unique image IDs were selected.
- Every selected JSON object is byte-semantically equal to the parent row with
  the same state ID; selection changed neither prompts nor targets.
- Every selected image resolves through the existing manifest loader.
- The selection report contains exactly 512 source/state audit rows and records
  `selection_fields=[source_id,state_id]`,
  `labels_used_for_ranking=false`, and
  `outcomes_used_for_ranking=false`.
- Target contents and existing Qwen-3B outcomes were not used to choose a
  source, state, hardware class, or analysis condition.

## Execution boundary

This activation authorizes only an engineering preflight and then the frozen
Qwen2.5-VL-7B opened-development diagnostic. Before GPU submission, the runner
must bind the protocol, activation, selected manifest, model revision, rollout
code, scorer, shard/merge code, and notification settings by hash. It must
re-query live quota and RTX 4090/H800/H100 queue state and follow the hardware
rule in the protocol.

ScreenQA calibration, formal, reserve, untouched, validation, and test roles
remain sealed. This population cannot select a deployment threshold or rescue
any earlier failed gate. No GitHub push is authorized by this activation.
