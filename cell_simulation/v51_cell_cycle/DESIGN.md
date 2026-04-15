# V51: DNA Replication & Cell Division

## The Cell Cycle

```
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 │
   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌──┴──┐
   │    G    │──────│    S    │──────│   G2    │──────│  M  │
   │ (Growth)│      │ (DNA    │      │ (Growth)│      │(Div)│
   │         │      │  Replic)│      │         │      │     │
   └─────────┘      └─────────┘      └─────────┘      └─────┘
       │                                                 │
       │              JCVI-syn3A                        │
       │           (no G1/G2 checkpoints)               │
       │                                                 │
       └─────────────────────────────────────────────────┘
```

## JCVI-syn3A Cell Cycle

Minimal cells have a SIMPLIFIED cell cycle:
- No distinct G1/G2 phases
- Replication initiates when cell reaches critical size
- Division follows replication completion
- ~60-90 min doubling time

## Key Molecular Players

### DNA Replication Initiation
- **DnaA**: Master regulator - initiates replication at origin
  - Binds ATP → active form
  - Accumulates during growth
  - Triggers initiation at threshold
  
### Replication Fork
- **DnaB**: Replicative helicase (unwinds DNA)
- **DnaC**: Helicase loader
- **DnaG**: Primase (makes RNA primers)
- **DnaE/DnaN/DnaX**: DNA Pol III (synthesizes DNA)
- **GyrA/GyrB**: Topoisomerase (relieves supercoiling)
- **LigA**: DNA ligase (seals fragments)

### Cell Division
- **FtsZ**: Tubulin homolog - forms Z-ring at midcell
- **FtsA**: Anchors Z-ring to membrane
- **FtsB/FtsL/FtsQ**: Early divisome
- **FtsW/FtsI**: Peptidoglycan synthesis
- **FtsK**: Chromosome segregation

## Implementation

### State Variables
```python
# Replication state
DNA_content = 1.0  # 1 = unreplicated, 2 = fully replicated
replication_progress = 0.0  # 0 to 1
replication_active = False

# Division state  
Z_ring_assembled = False
division_progress = 0.0  # 0 to 1
```

### Initiation Logic
```python
# DnaA-ATP triggers initiation
DnaA_ATP = DnaA * ATP / (Kd + ATP)
if DnaA_ATP > threshold and not replicating:
    start_replication()
```

### Replication Progress
```python
# Replication rate depends on:
# - dNTP availability
# - DNA Pol III levels
# - Helicase/primase levels
v_replication = k_rep * [DnaE] * [dNTPs] / (Km + [dNTPs])
```

### Division Trigger
```python
# FtsZ polymerization triggers division
if DNA_content >= 2.0 and [FtsZ] > threshold:
    assemble_Z_ring()
    
if Z_ring_assembled:
    progress_division()
```

### Division Event
```python
if division_progress >= 1.0:
    # Split everything in half
    for metabolite in metabolites:
        metabolite /= 2
    for protein in proteins:
        protein /= 2
    DNA_content = 1.0
    reset_cell_cycle()
```

## Emergent Cell Cycle

The beautiful thing: cell cycle timing EMERGES from:
1. Growth rate (ribosome content)
2. DnaA accumulation rate
3. dNTP synthesis capacity
4. FtsZ expression level

No explicit timers needed!
