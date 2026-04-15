"""
PPI-based features for gene essentiality prediction.
Data from SynWiki (synwiki.uni-goettingen.de)
"""
import csv
from collections import defaultdict
from pathlib import Path

# Load PPI network
PPI_FILE = Path(__file__).parent / "synwiki_ppi.csv"

def load_ppi_network():
    """Load PPI network as adjacency dict."""
    network = defaultdict(set)
    
    with open(PPI_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g1 = row['protein1_locus']
            g2 = row['protein2_locus']
            network[g1].add(g2)
            network[g2].add(g1)
    
    return dict(network)

# Precompute
_PPI_NETWORK = None

def get_ppi_network():
    global _PPI_NETWORK
    if _PPI_NETWORK is None:
        _PPI_NETWORK = load_ppi_network()
    return _PPI_NETWORK

def ppi_degree(gene: str) -> int:
    """Number of interaction partners."""
    network = get_ppi_network()
    return len(network.get(gene, set()))

def ppi_in_network(gene: str) -> bool:
    """Whether gene has any known interactions."""
    return gene in get_ppi_network()

def ppi_neighbors_essential(gene: str, essentials: set) -> float:
    """Fraction of neighbors that are essential."""
    network = get_ppi_network()
    neighbors = network.get(gene, set())
    if not neighbors:
        return 0.5  # Unknown
    return sum(1 for n in neighbors if n in essentials) / len(neighbors)

def get_ppi_features(gene: str, essentials: set = None) -> dict:
    """Get all PPI-based features for a gene."""
    if essentials is None:
        essentials = set()
    
    return {
        'ppi_degree': ppi_degree(gene),
        'ppi_in_network': 1 if ppi_in_network(gene) else 0,
        'ppi_neighbor_ess_frac': ppi_neighbors_essential(gene, essentials),
    }

if __name__ == "__main__":
    network = get_ppi_network()
    print(f"Loaded {len(network)} genes with interactions")
    print(f"Total edges: {sum(len(v) for v in network.values()) // 2}")
    
    # Top connected
    by_degree = sorted(network.items(), key=lambda x: -len(x[1]))
    print("\nTop connected genes:")
    for g, partners in by_degree[:10]:
        print(f"  {g}: {len(partners)} partners")
