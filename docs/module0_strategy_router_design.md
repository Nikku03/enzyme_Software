# Module 0 - Strategy Router (Conceptual Design v1)

## Purpose
Module 0 is the decision brain of BondBreak v1. It does not design enzymes, run simulations, or predict wet-lab success. Its sole purpose is to decide whether a job is worth doing and, if so, how to do it cheaply and correctly. Think of Module 0 as a scientific triage system.

## Why Module 0 exists
Without Module 0, the pipeline would:
- Run expensive tunnel analysis on impossible chemistry
- Attempt hydrolase designs for C-H bonds
- Waste MD on jobs that violate pH or metal constraints
- Scale poorly and lose credibility

With Module 0:
- 30-50% of bad jobs die in minutes
- The right chemistry is used from the start
- Costs and timelines become predictable
- Clients trust the system because it says NO when needed

Module 0 is what turns BondBreak from "AI enzyme generator" into a biophysical logic engine.

## Inputs
1) Substrate definition
- Molecular structure (SMILES / SDF)
- Explicit bond to break

2) Operational constraints (optional)
- pH window
- temperature
- metal tolerance
- oxidation allowed or forbidden
- host organism

These constraints are hard physics, not preferences.

## Internal workflow

Step 1 - Molecular sanity check
- Parse the molecule
- Ensure the bond exists
- Check basic structural validity

This prevents garbage-in scenarios.

Step 2 - Bond context interpretation
- Bond type (ester, amide, C-H, aromatic)
- Whether the bond is activated or inert
- Nearby functional groups
- Ring context
- Polarity vs non-polarity

This step answers: "What kind of chemistry is this, really?"

Step 3 - Symmetry recognition
If the molecule has equivalent bonds (e.g., multiple identical methyl C-H bonds):
- Detect symmetry
- Group equivalent bonds together
- Treat them as one design problem

Why this matters:
- Avoids redundant computation
- Prevents user confusion
- Matches how chemistry actually works

Step 4 - Difficulty assessment
Module 0 assigns a difficulty tier:
- EASY: polar, well-known chemistry (esters, phosphates)
- MEDIUM: constrained polar bonds, unusual environments
- HARD: C-H bonds, buried targets, radical chemistry

This classification directly controls:
- How much compute is spent
- How many designs are generated
- How many MD simulations are run
- What success rate is promised

Step 5 - Mechanism routing (expert selection)
Based on bond type and constraints, Module 0 selects:
- Plausible reaction mechanisms
- Required cofactors
- Compatible enzyme families

Examples:
- Ester -> serine hydrolase
- Amide -> amidase / metalloenzyme
- C-H -> P450-like or radical SAM

This step prevents mechanistic mismatches, the #1 cause of silent failure in enzyme design.

Step 6 - Constraint viability check (light solvent gate)
Module 0 checks for obvious physical contradictions:
- C-H activation but metals forbidden -> impossible
- Oxidation required but oxidation forbidden -> impossible
- Extreme pH -> unstable catalytic residues

These are cheap kill checks that save days later.

Step 7 - Confidence estimation (router confidence)
Module 0 assigns a router confidence score (0-1). This is not a probability of success, but confidence that the chosen route makes sense. The score is used to:
- Flag LOW_CONF jobs
- Recommend human review
- Later integrate into 2027 causal probability chains

Step 8 - GO / LOW_CONF / NO_GO decision
Module 0 outputs one of three states:
- GO: chemistry is feasible, proceed automatically
- LOW_CONF: plausible but risky, proceed with caution or review
- NO_GO: violates chemistry or constraints, stop immediately

This is scientific honesty and a competitive advantage.

Step 9 - Compute plan generation
If GO or LOW_CONF, Module 0 defines:
- How many scaffolds to test
- How strict TopoGate should be
- Whether QM is needed
- How much MD to run (or skip)

This yields 10-100x speedup without cheating physics.

Step 10 - Failure logging (if stopped)
If a job is rejected or marked low confidence:
- The reason is logged
- The conditions are logged
- The bond context is logged

This builds a negative knowledge base that:
- Improves routing over time
- Feeds directly into the 2027 causal engine
- Is usually discarded by competitors

## Outputs
Module 0 emits a Job Card that acts as a contract for the rest of the pipeline. It contains:
- Bond interpretation
- Equivalent bond mapping
- Mechanism choice
- Difficulty tier
- Confidence score
- Compute budget
- Explicit reasoning

Every downstream module simply executes the plan.

## System impact
Without Module 0:
- Chaotic compute
- Poor scalability
- Unexplained failures
- Low trust from clients

With Module 0:
- Predictable cost and time
- Early rejection of impossible jobs
- Higher wet-lab hit rates
- Clean audit trail of decisions
- Smooth path to causal discovery

## Why Module 0 is foundational for 2027
Module 0 already:
- Reasons causally
- Handles uncertainty explicitly
- Rejects weak hypotheses
- Logs failure modes

The only difference between v1 and 2027 is scale, not philosophy.

## Summary
Module 0 is a causality-aware routing layer that decides whether, how, and how much enzyme design should be attempted for a given bond-breaking problem. It enforces chemical realism, respects constraints, eliminates redundant work, and turns vague design requests into a structured, auditable execution plan. By rejecting failure early and choosing the correct mechanistic path, Module 0 enables BondBreak v1 to be fast, affordable, honest, and scalable, while quietly laying the foundation for a full biological discovery engine in 2027.
