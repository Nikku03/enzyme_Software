# Dark Manifold: UPDATED Project Plan
## With Reusable Components from enzyme_Software

**Updated:** Incorporating 821 Python files of existing work

---

## REUSABLE COMPONENTS INVENTORY

### ⭐⭐⭐⭐⭐ HIGH VALUE - Direct Reuse

| Component | Location | Lines | What It Does | Adaptation for Cells |
|-----------|----------|-------|--------------|---------------------|
| **HyperbolicMemoryBank** | `nexus/reasoning/hyperbolic_memory.py` | 1,326 | Poincaré ball retrieval + PGW transport | Replace molecules → genes |
| **MechanismEncoder** | `nexus/reasoning/metric_learner.py` | 322 | Fingerprint → 128-dim embedding | Replace fingerprints → gene features |
| **PGWTransporter** | `nexus/reasoning/pgw_transport.py` | ~400 | Partial Gromov-Wasserstein transport | Transfer essentiality between similar genes |
| **RuleDiscoveryModule** | `src/enzyme_software/liquid_nn_v2/model/dark_manifold.py` | ~100 | Extract interpretable rules from predictions | "Hub genes are essential" rules |
| **TopologyFeatures** | `src/enzyme_software/liquid_nn_v2/features/topology_features.py` | ~80 | Graph centrality, scaffold membership | Gene network position features |

### ⭐⭐⭐ MEDIUM VALUE - Partial Reuse

| Component | Location | What It Does | Adaptation |
|-----------|----------|--------------|------------|
| **HGNNProjection** | `nexus/reasoning/metric_learner.py` | Project features → Poincaré ball | Use for gene embeddings |
| **PoincareMath** | `nexus/reasoning/metric_learner.py` | Safe hyperbolic operations | Reuse directly |
| **BaselineMemoryBank** | `nexus/reasoning/baseline_memory.py` | Simple cosine similarity retrieval | Simpler fallback option |
| **PGA (16D)** | `nexus/pocket/pga.py` | Projective geometric algebra | Could encode pathway geometry |

### ⭐⭐ LOWER VALUE - Concepts Only

| Component | Location | What It Does | Notes |
|-----------|----------|--------------|-------|
| **ContinuousField (SIREN)** | Multiple locations | Neural implicit representation | Overkill for this project |
| **DarkMatterField** | `dark_manifold.py` | Latent field dynamics | Cool but complex |
| **FieldDynamics** | `dark_manifold.py` | Neural ODE evolution | Save for future work |

---

## REVISED ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: NEURAL REFINEMENT                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Gene Encoder    │  │ Hyperbolic Memory │  │ Rule Discovery    │  │
│  │ (from Mechanism │  │ (from HyperbolicMB│  │ (interpretable    │  │
│  │  Encoder)       │  │  + PGW Transport) │  │  essentiality     │  │
│  │                 │  │                   │  │  rules)           │  │
│  │ Features:       │  │ Retrieve similar  │  │                   │  │
│  │ - Centrality    │  │ genes, transfer   │  │ "If hub AND no    │  │
│  │ - Isozymes      │  │ essentiality      │  │  isozyme → 95%    │  │
│  │ - Expression    │  │ predictions       │  │  essential"       │  │
│  │ - Pathway       │  │                   │  │                   │  │
│  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │
│           │                    │                      │             │
│           └────────────────────┼──────────────────────┘             │
│                                ▼                                    │
│                    ┌───────────────────────┐                        │
│                    │   Combined Predictor   │                        │
│                    │   FBA + Memory + Rules │                        │
│                    │   → 90%+ accuracy      │                        │
│                    └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: FBA CORE (V37)                          │
│                    85.6% accuracy baseline                          │
│                    DON'T TOUCH                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## UPDATED IMPLEMENTATION PLAN

### Phase 1: Setup & Consolidation (4 days)

#### Task 1.1: Create Package Structure
```
dark_manifold/
├── __init__.py
├── data/
│   ├── imb155.py              # From V37
│   ├── essentiality.py        # Hutchison 2016
│   └── gene_features.py       # NEW: gene feature extraction
├── models/
│   ├── fba.py                 # From V37 (untouched)
│   ├── gene_encoder.py        # ADAPT from MechanismEncoder
│   ├── memory_bank.py         # ADAPT from HyperbolicMemoryBank
│   ├── rule_discovery.py      # ADAPT from RuleDiscoveryModule
│   └── combined_predictor.py  # NEW: FBA + Memory + Rules
├── reasoning/                  # COPY from nexus/reasoning/
│   ├── hyperbolic_memory.py   # Copy + adapt
│   ├── metric_learner.py      # Copy + adapt
│   ├── pgw_transport.py       # Copy directly
│   └── poincare_math.py       # Extract from metric_learner
└── validation/
    └── benchmark.py           # NEW
```

#### Task 1.2: Copy and Adapt Core Components

**From `nexus/reasoning/metric_learner.py`:**
```python
# Original: Morgan fingerprints → embedding
class MechanismEncoder(nn.Module):
    def __init__(self, fp_bits: int = 2048, embed_dim: int = 128):
        ...

# Adapted: Gene features → embedding  
class GeneEncoder(nn.Module):
    def __init__(self, feature_dim: int = 15, embed_dim: int = 128):
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), p=2, dim=-1)
```

**From `nexus/reasoning/hyperbolic_memory.py`:**
```python
# Original: Molecules with SoM labels
class HyperbolicMemoryBank:
    def store(self, mol, som_idx: int): ...
    def retrieve_and_transport(self, query_mol): ...

# Adapted: Genes with essentiality labels
class GeneMemoryBank:
    def store(self, gene: str, features: np.ndarray, essential: bool): ...
    def retrieve_similar(self, query_features: np.ndarray, k: int = 5): ...
    def transfer_essentiality(self, query: str, retrieved: str): ...
```

---

### Phase 2: Gene Feature Engineering (3 days)

#### Task 2.1: Define Gene Features

```python
# dark_manifold/data/gene_features.py

def extract_gene_features(gene: str, model: FBAModel) -> np.ndarray:
    """
    Extract features for neural refinement.
    
    Combines:
    - FBA results (biomass impact)
    - Network topology (from existing TopologyFeatures concept)
    - Expression data (from V47)
    - Pathway context
    """
    # Run FBA knockout
    fba_result = model.knockout(gene)
    
    # Get network topology (ADAPT from topology_features.py)
    topology = compute_gene_topology(gene)
    
    # Get expression data (from V47)
    expression = get_expression_level(gene)
    
    return np.array([
        # FBA features (3)
        fba_result['biomass_ratio'],
        fba_result['num_blocked_reactions'],
        fba_result['flux_variability'],
        
        # Network topology (5) - adapted from TopologyFeatures
        topology['degree_centrality'],
        topology['betweenness_centrality'],
        topology['closeness_centrality'],
        topology['is_hub'],  # degree > mean + 2*std
        topology['clustering_coefficient'],
        
        # Redundancy features (3)
        count_isozymes(gene),
        has_alternative_pathway(gene),
        pathway_redundancy_score(gene),
        
        # Expression features (2) - from V47
        expression['level'],
        expression['protein_halflife'],
        
        # Thermodynamic features (2) - from V38
        get_reaction_delta_g(gene),
        is_irreversible(gene),
    ])

FEATURE_DIM = 15
```

#### Task 2.2: Build Gene Network Graph

```python
# dark_manifold/data/gene_network.py

import networkx as nx

def build_metabolic_network(stoichiometry: np.ndarray, genes: list) -> nx.DiGraph:
    """
    Build gene-gene interaction network from stoichiometry.
    
    Two genes are connected if they share a metabolite
    (product of one is substrate of another).
    """
    G = nx.DiGraph()
    
    # Add all genes as nodes
    for gene in genes:
        G.add_node(gene)
    
    # Add edges based on metabolite sharing
    for i, gene_i in enumerate(genes):
        for j, gene_j in enumerate(genes):
            if i != j:
                if shares_metabolite(gene_i, gene_j, stoichiometry):
                    G.add_edge(gene_i, gene_j)
    
    return G

def compute_gene_topology(gene: str, G: nx.DiGraph) -> dict:
    """
    Compute topology features for a gene.
    
    ADAPTED from: topology_features.py compute_atom_topology_features()
    """
    return {
        'degree_centrality': nx.degree_centrality(G)[gene],
        'betweenness_centrality': nx.betweenness_centrality(G)[gene],
        'closeness_centrality': nx.closeness_centrality(G)[gene],
        'is_hub': G.degree(gene) > np.mean([G.degree(n) for n in G.nodes()]) + 2*np.std(...),
        'clustering_coefficient': nx.clustering(G, gene),
    }
```

---

### Phase 3: Memory-Augmented Prediction (5 days)

#### Task 3.1: Adapt HyperbolicMemoryBank

```python
# dark_manifold/reasoning/gene_memory.py

class GeneMemoryBank:
    """
    Adapted from HyperbolicMemoryBank for gene essentiality.
    
    Key changes:
    - Molecules → Genes
    - Morgan fingerprints → Gene feature vectors
    - SoM labels → Essentiality labels
    - MCS transport → Network distance transport
    """
    
    def __init__(
        self,
        device: str = "cpu",
        curvature: float = 1.0,
        feature_dim: int = 15,
        embed_dim: int = 128,
        poincare_radius: float = 0.95,
    ):
        self.encoder = GeneEncoder(feature_dim, embed_dim)
        self.poincare = PoincareMath(c=curvature)
        
        # Memory storage
        self.gene_features: List[np.ndarray] = []
        self.gene_names: List[str] = []
        self.essentiality: List[bool] = []
        self.embeddings: List[torch.Tensor] = []
    
    def store(self, gene: str, features: np.ndarray, essential: bool):
        """Add a gene to memory."""
        self.gene_names.append(gene)
        self.gene_features.append(features)
        self.essentiality.append(essential)
        
        # Compute Poincaré embedding
        with torch.no_grad():
            embed = self.encoder(torch.tensor(features))
            poincare_embed = self.poincare.exp_map_0(embed)
            self.embeddings.append(poincare_embed)
    
    def retrieve(self, query_features: np.ndarray, k: int = 5) -> List[dict]:
        """
        Find k most similar genes in hyperbolic space.
        
        Uses geodesic distance in Poincaré ball (not Euclidean!).
        """
        query_embed = self.encoder(torch.tensor(query_features))
        query_poincare = self.poincare.exp_map_0(query_embed)
        
        # Compute hyperbolic distances
        distances = []
        for stored_embed in self.embeddings:
            d = self.poincare.geodesic_distance(query_poincare, stored_embed)
            distances.append(d.item())
        
        # Get top-k closest
        top_k_idx = np.argsort(distances)[:k]
        
        return [
            {
                'gene': self.gene_names[i],
                'distance': distances[i],
                'essential': self.essentiality[i],
                'features': self.gene_features[i],
            }
            for i in top_k_idx
        ]
    
    def predict_by_analogy(self, query_features: np.ndarray) -> dict:
        """
        Predict essentiality by retrieving similar genes.
        
        Weighted vote based on hyperbolic distance.
        """
        neighbors = self.retrieve(query_features, k=5)
        
        # Distance-weighted voting
        weights = [1.0 / (n['distance'] + 0.01) for n in neighbors]
        total_weight = sum(weights)
        
        essential_score = sum(
            w * (1.0 if n['essential'] else 0.0)
            for w, n in zip(weights, neighbors)
        ) / total_weight
        
        return {
            'essential_score': essential_score,
            'confidence': 1.0 - np.std([n['essential'] for n in neighbors]),
            'neighbors': neighbors,
        }
```

#### Task 3.2: Adapt Rule Discovery

```python
# dark_manifold/reasoning/gene_rules.py

class GeneRuleDiscovery:
    """
    Adapted from RuleDiscoveryModule for gene essentiality.
    
    Learns interpretable rules like:
    - "Hub genes with no isozymes are essential"
    - "Genes in linear pathways are essential"
    - "Highly expressed genes are more likely essential"
    """
    
    def __init__(self, feature_dim: int = 15, num_rules: int = 16):
        self.feature_dim = feature_dim
        self.num_rules = num_rules
        
        # Rule prototypes (learned)
        self.rule_centers = nn.Parameter(torch.randn(num_rules, feature_dim))
        self.rule_outputs = nn.Parameter(torch.zeros(num_rules))  # essentiality bias
        self.rule_confidence = nn.Parameter(torch.zeros(num_rules))
        
        # Rule statistics (accumulated)
        self.rule_counts = torch.zeros(num_rules)
        self.rule_accuracy = torch.zeros(num_rules)
    
    def match_rules(self, features: torch.Tensor) -> dict:
        """Find which rules match the gene."""
        # Cosine similarity to rule prototypes
        similarities = F.cosine_similarity(
            features.unsqueeze(0),
            self.rule_centers,
            dim=-1
        )
        
        # Soft assignment
        weights = F.softmax(similarities * 5.0, dim=-1)
        
        # Weighted prediction
        prediction = (weights * torch.sigmoid(self.rule_outputs)).sum()
        
        return {
            'prediction': prediction,
            'rule_weights': weights,
            'top_rule': similarities.argmax().item(),
            'confidence': torch.sigmoid(self.rule_confidence[similarities.argmax()]).item(),
        }
    
    def interpret_rules(self) -> List[str]:
        """
        Generate human-readable rule descriptions.
        
        Looks at which features have highest weight in each rule prototype.
        """
        FEATURE_NAMES = [
            'biomass_ratio', 'blocked_reactions', 'flux_variability',
            'degree_centrality', 'betweenness', 'closeness', 'is_hub', 'clustering',
            'isozyme_count', 'alt_pathway', 'pathway_redundancy',
            'expression', 'protein_halflife',
            'delta_g', 'irreversible',
        ]
        
        rules = []
        for i in range(self.num_rules):
            center = self.rule_centers[i].detach().numpy()
            output = torch.sigmoid(self.rule_outputs[i]).item()
            confidence = torch.sigmoid(self.rule_confidence[i]).item()
            
            if confidence < 0.3:
                continue  # Skip low-confidence rules
            
            # Find top 2 features
            top_features = np.argsort(np.abs(center))[-2:]
            
            conditions = []
            for f_idx in top_features:
                f_name = FEATURE_NAMES[f_idx]
                f_value = center[f_idx]
                if f_value > 0.5:
                    conditions.append(f"high {f_name}")
                elif f_value < -0.5:
                    conditions.append(f"low {f_name}")
            
            outcome = "ESSENTIAL" if output > 0.5 else "NON-ESSENTIAL"
            
            rule_str = f"IF {' AND '.join(conditions)} THEN {outcome} ({confidence:.0%} conf)"
            rules.append(rule_str)
        
        return rules
```

---

### Phase 4: Combined Predictor (3 days)

```python
# dark_manifold/models/combined_predictor.py

class CombinedEssentialityPredictor:
    """
    Combines:
    1. FBA (V37) - 85.6% baseline
    2. Memory-based prediction (similar genes)
    3. Rule-based prediction (learned patterns)
    
    Final prediction is weighted combination.
    """
    
    def __init__(
        self,
        fba_model: FBAModel,
        memory_bank: GeneMemoryBank,
        rule_module: GeneRuleDiscovery,
        fba_weight: float = 0.5,
        memory_weight: float = 0.3,
        rule_weight: float = 0.2,
    ):
        self.fba = fba_model
        self.memory = memory_bank
        self.rules = rule_module
        
        self.weights = {
            'fba': fba_weight,
            'memory': memory_weight,
            'rules': rule_weight,
        }
    
    def predict(self, gene: str) -> dict:
        """
        Predict essentiality using all three methods.
        """
        # 1. FBA prediction
        fba_result = self.fba.knockout(gene)
        fba_score = 1.0 - fba_result['biomass_ratio']  # Higher = more essential
        
        # 2. Extract features
        features = extract_gene_features(gene, self.fba)
        
        # 3. Memory prediction
        memory_result = self.memory.predict_by_analogy(features)
        memory_score = memory_result['essential_score']
        
        # 4. Rule prediction
        rule_result = self.rules.match_rules(torch.tensor(features))
        rule_score = rule_result['prediction'].item()
        
        # 5. Weighted combination
        combined_score = (
            self.weights['fba'] * fba_score +
            self.weights['memory'] * memory_score +
            self.weights['rules'] * rule_score
        )
        
        # 6. Confidence from agreement
        scores = [fba_score, memory_score, rule_score]
        agreement = 1.0 - np.std(scores)
        
        return {
            'essential': combined_score > 0.5,
            'score': combined_score,
            'confidence': agreement,
            'fba_score': fba_score,
            'memory_score': memory_score,
            'rule_score': rule_score,
            'similar_genes': memory_result['neighbors'],
            'top_rule': rule_result['top_rule'],
            'explanation': self._explain(gene, fba_result, memory_result, rule_result),
        }
    
    def _explain(self, gene, fba, memory, rules) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        # FBA explanation
        if fba['biomass_ratio'] < 0.1:
            parts.append(f"FBA: Knockout blocks {fba['num_blocked_reactions']} reactions")
        
        # Memory explanation
        if memory['neighbors']:
            similar = memory['neighbors'][0]
            parts.append(f"Similar to {similar['gene']} ({'essential' if similar['essential'] else 'non-essential'})")
        
        # Rule explanation
        if rules['confidence'] > 0.5:
            parts.append(f"Matches rule #{rules['top_rule']}")
        
        return " | ".join(parts)
```

---

## REVISED TIMELINE

| Phase | Task | Reused From | Time |
|-------|------|-------------|------|
| **1** | Package structure | - | 1 day |
| **1** | Copy FBA core | V37 | 0.5 days |
| **1** | Copy reasoning modules | nexus/reasoning/ | 1 day |
| **1** | Adapt MechanismEncoder → GeneEncoder | metric_learner.py | 1 day |
| **2** | Gene feature extraction | topology_features.py | 1.5 days |
| **2** | Gene network graph | - | 1.5 days |
| **3** | Adapt HyperbolicMemoryBank | hyperbolic_memory.py | 2 days |
| **3** | Adapt RuleDiscoveryModule | dark_manifold.py | 1.5 days |
| **3** | PGW transport for genes | pgw_transport.py | 1.5 days |
| **4** | Combined predictor | - | 2 days |
| **4** | Training pipeline | - | 1 day |
| **5** | Synthetic lethality | - | 4 days |
| **6** | External validation | - | 3 days |
| **7** | Paper | - | 7 days |

**TOTAL: ~28 days (down from 38!)**

---

## WHAT THIS GIVES YOU

### Unique Selling Points for Paper

1. **Memory-Augmented Prediction**
   - "First use of hyperbolic memory banks for gene essentiality"
   - Retrieves similar genes and transfers predictions
   - Explainable: "Predicted essential because similar to dnaK, groEL"

2. **Interpretable Rules**
   - "Discovered 12 rules with >80% accuracy"
   - Example: "IF hub gene AND no isozyme THEN essential (92% conf)"
   - Biologically meaningful insights

3. **Multi-Method Ensemble**
   - FBA (physics) + Memory (analogy) + Rules (learned patterns)
   - Better than any single method

4. **Novel Architecture**
   - Hyperbolic geometry for gene similarity (natural for hierarchical data)
   - PGW transport adapted from chemistry to biology

---

## NEXT STEPS

1. **Today:** Create package structure, copy reasoning modules
2. **Tomorrow:** Adapt GeneEncoder from MechanismEncoder
3. **Day 3:** Implement gene feature extraction
4. **Day 4:** Start adapting HyperbolicMemoryBank

Want me to start implementing Phase 1 now?
