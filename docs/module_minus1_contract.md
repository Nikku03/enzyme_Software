Module -1 Contract (Reactivity Hub)

Version: `module_minus1.v1`

Module -1 runs before Module 0 and emits a stable handoff contract for target-bond resolution and reactivity priors.

Shared IO input contract (always present)
- `shared_io.input.substrate_context`:
  - `smiles`
  - `mol_block` (optional)
- `shared_io.input.bond_spec`:
  - `target_bond`
  - `target_bond_indices` (optional)
  - `selection_mode`
  - `resolved_target` (always present; can be a no-match stub)
  - `context`
- `shared_io.input.condition_profile`
- `shared_io.input.telemetry`:
  - `run_id`
  - `trace`
  - `warnings`

Module -1 output namespace
- `shared_io.outputs.module_minus1.result`
- `shared_io.outputs.module_minus1.sre_atr`
- `shared_io.outputs.module_minus1.fragment_builder`
- `shared_io.outputs.module_minus1.cpt`
- `shared_io.outputs.module_minus1.level1`
- `shared_io.outputs.module_minus1.level2`
- `shared_io.outputs.module_minus1.level3`
- `shared_io.outputs.module_minus1.ep_av`
- `shared_io.outputs.module_minus1.evidence_record`

Required `resolved_target` fields
- `requested`
- `selection_mode`
- `match_count`
- `candidate_bonds` (list, can be empty)
- `next_input_required` (e.g., `["target_bond"]` on no match)

Downstream behavior
- Module 0 reads Module -1 competition/equivalent-site signals.
- Equivalent best sites trigger disambiguation handling (`HALT_NEED_SELECTION` / target clarification).
- High-confidence Module -1 PASS can provide a small route-confidence uplift.

Extension guidance
- Add new Module -1 outputs only under `shared_io.outputs.module_minus1.*`.
- Keep existing top-level keys in `module_minus1_sre` backward-compatible.
- If adding new constraints, include them under `module_minus1_schema.constraint_flags`.
