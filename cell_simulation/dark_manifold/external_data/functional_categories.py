"""
Functional category features from SynWiki.
Categories indicate biological role - strong essentiality signal.
"""

# From SynWiki category browser
# Genes in information processing / translation / metabolism are likely essential

CATEGORY_ESSENTIAL_WEIGHT = {
    # High essentiality
    'Translation': 0.95,
    'tRNA': 0.95,
    'Ribosomal proteins': 0.95,
    'Ribosomal RNA': 0.95,
    'DNA replication': 0.90,
    'Transcription': 0.85,
    'Aminoacyl-tRNA synthase': 0.95,
    'ATP synthesis': 0.85,
    'Carbon core metabolism': 0.80,
    'Lipid metabolism': 0.75,
    'Cofactor acquisition': 0.70,
    'Protein secretion': 0.70,
    'Chaperones': 0.70,
    
    # Medium essentiality
    'Transporters': 0.60,
    'ABC transporters': 0.60,
    'ECF transporters': 0.65,
    'Phosphotransferase system': 0.70,
    'DNA repair': 0.55,
    'Homeostasis': 0.50,
    
    # Lower essentiality
    'Proteins of unknown function': 0.50,
    'Foreign gene': 0.30,
    'Pseudogenes': 0.10,
}

# Known functional annotations from SynWiki essential genes page
SYNWIKI_FUNCTIONS = {
    # From essential genes list - these are confirmed essential
    'JCVISYN3A_0001': 'DNA replication',  # dnaA
    'JCVISYN3A_0002': 'DNA replication',  # dnaN
    'JCVISYN3A_0006': 'DNA topology',     # gyrB
    'JCVISYN3A_0007': 'DNA topology',     # gyrA
    'JCVISYN3A_0030': 'Transport',
    'JCVISYN3A_0143': 'Unclear',
    'JCVISYN3A_0146': 'Unclear',
    'JCVISYN3A_0281': 'Unclear',
    'JCVISYN3A_0296': 'Unclear',
    'JCVISYN3A_0317': 'Unclear',  # FN - non-metabolic essential
    'JCVISYN3A_0352': 'Unclear',  # FN - quasi-essential
    'JCVISYN3A_0353': 'Cell division',  # GpsB - FN
    'JCVISYN3A_0373': 'Unclear',
    'JCVISYN3A_0379': 'Unclear',
    'JCVISYN3A_0388': 'Unclear',
    'JCVISYN3A_0233': 'PTS',  # PtsI - FN
    'JCVISYN3A_0207': 'Translation',  # FN
    'JCVISYN3A_0234': 'PTS',  # Crr
    'JCVISYN3A_0522': 'Cell division',  # FtsZ
    'JCVISYN3A_0239': 'Cell division',  # EzrA
    
    # Non-essential (FP genes)
    'JCVISYN3A_0683': 'Unknown',  # FP
    'JCVISYN3A_0684': 'Unknown',  # FP  
    'JCVISYN3A_0589': 'Unknown',  # FP
    'JCVISYN3A_0235': 'Unknown',  # FP
}

def get_function_category(gene: str) -> str:
    """Get functional category for a gene."""
    return SYNWIKI_FUNCTIONS.get(gene, 'Unknown')

def get_category_essentiality_prior(gene: str) -> float:
    """Get prior probability of essentiality based on function."""
    func = get_function_category(gene)
    for cat, weight in CATEGORY_ESSENTIAL_WEIGHT.items():
        if cat.lower() in func.lower():
            return weight
    return 0.5  # Unknown

if __name__ == "__main__":
    print("Function category essentiality priors:")
    for gene, func in list(SYNWIKI_FUNCTIONS.items())[:10]:
        prior = get_category_essentiality_prior(gene)
        print(f"  {gene}: {func} -> {prior:.2f}")
