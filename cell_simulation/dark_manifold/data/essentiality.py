"""
dark_manifold/data/essentiality.py

Experimental gene essentiality data from Hutchison et al. 2016 Science.
Transposon mutagenesis results for JCVI-syn3A.

Labels:
- E = Essential (knockout is lethal)
- Q = Quasi-essential (severely impaired growth)
- N = Non-essential (viable knockout)

For prediction purposes, we treat E and Q as "essential" (positive class).
"""

from typing import Dict, Tuple, List

# ============================================================================
# EXPERIMENTAL GENE ESSENTIALITY DATA
# From Hutchison et al. 2016 Science - Transposon mutagenesis
# ============================================================================

GENE_ESSENTIALITY: Dict[str, str] = {
    # Glycolysis - mostly essential
    'JCVISYN3A_0685': 'E',  # ptsG - glucose PTS
    'JCVISYN3A_0233': 'E',  # pgi - phosphoglucose isomerase
    'JCVISYN3A_0207': 'E',  # pfkA - phosphofructokinase
    'JCVISYN3A_0352': 'E',  # fba - fructose-bisphosphate aldolase
    'JCVISYN3A_0353': 'E',  # tpiA - triose phosphate isomerase
    'JCVISYN3A_0314': 'E',  # gapA - glyceraldehyde-3P dehydrogenase
    'JCVISYN3A_0315': 'E',  # pgk - phosphoglycerate kinase
    'JCVISYN3A_0689': 'E',  # pgm - phosphoglycerate mutase
    'JCVISYN3A_0231': 'E',  # eno - enolase
    'JCVISYN3A_0546': 'E',  # pyk - pyruvate kinase
    
    # Fermentation - non-essential
    'JCVISYN3A_0449': 'N',  # ldh - lactate dehydrogenase
    'JCVISYN3A_0589': 'N',  # pfl - pyruvate formate lyase
    'JCVISYN3A_0484': 'N',  # pta - phosphotransacetylase
    'JCVISYN3A_0485': 'N',  # ackA - acetate kinase
    
    # Pentose phosphate pathway
    'JCVISYN3A_0439': 'Q',  # zwf - glucose-6P dehydrogenase
    'JCVISYN3A_0440': 'Q',  # pgl - 6-phosphogluconolactonase
    'JCVISYN3A_0441': 'E',  # gnd - 6-phosphogluconate dehydrogenase
    'JCVISYN3A_0509': 'E',  # rpe - ribulose-5P epimerase
    'JCVISYN3A_0510': 'E',  # rpi - ribose-5P isomerase
    'JCVISYN3A_0234': 'E',  # tkt - transketolase
    'JCVISYN3A_0235': 'N',  # tal - transaldolase
    
    # Nucleotide synthesis
    'JCVISYN3A_0317': 'E',  # prsA - PRPP synthetase
    'JCVISYN3A_0416': 'E',  # ndk - nucleoside diphosphate kinase
    'JCVISYN3A_0005': 'E',  # adk - adenylate kinase
    'JCVISYN3A_0629': 'E',  # gmk - guanylate kinase
    'JCVISYN3A_0381': 'E',  # cmk - CMP/UMP kinase
    'JCVISYN3A_0536': 'E',  # tmk - thymidylate kinase
    'JCVISYN3A_0537': 'E',  # thyA - thymidylate synthase
    'JCVISYN3A_0319': 'E',  # nrdE - ribonucleotide reductase
    'JCVISYN3A_0320': 'E',  # nrdF - ribonucleotide reductase
    
    # Energy metabolism (ATP synthase)
    'JCVISYN3A_0783': 'E',  # atpA - ATP synthase alpha
    'JCVISYN3A_0782': 'E',  # atpB - ATP synthase beta
    'JCVISYN3A_0784': 'E',  # atpC - ATP synthase gamma
    'JCVISYN3A_0785': 'E',  # atpD - ATP synthase delta
    'JCVISYN3A_0786': 'E',  # atpE - ATP synthase c
    'JCVISYN3A_0787': 'E',  # atpF - ATP synthase b
    'JCVISYN3A_0788': 'E',  # atpG - ATP synthase a
    'JCVISYN3A_0789': 'E',  # atpH - ATP synthase epsilon
    
    # Replication
    'JCVISYN3A_0001': 'E',  # dnaA - replication initiation
    'JCVISYN3A_0690': 'E',  # dnaE - DNA polymerase III alpha
    'JCVISYN3A_0002': 'E',  # dnaN - DNA pol III beta clamp
    'JCVISYN3A_0192': 'E',  # dnaX - DNA pol III gamma/tau
    'JCVISYN3A_0643': 'Q',  # polA - DNA polymerase I
    'JCVISYN3A_0377': 'E',  # ligA - DNA ligase
    'JCVISYN3A_0691': 'E',  # dnaG - primase
    'JCVISYN3A_0692': 'E',  # dnaB - helicase
    'JCVISYN3A_0693': 'E',  # ssb - single-strand binding
    'JCVISYN3A_0694': 'E',  # gyrA - DNA gyrase A
    'JCVISYN3A_0695': 'E',  # gyrB - DNA gyrase B
    
    # Transcription
    'JCVISYN3A_0790': 'E',  # rpoA - RNA pol alpha
    'JCVISYN3A_0218': 'E',  # rpoB - RNA pol beta
    'JCVISYN3A_0217': 'E',  # rpoC - RNA pol beta'
    'JCVISYN3A_0792': 'E',  # rpoD - sigma factor
    'JCVISYN3A_0793': 'N',  # rpoE - sigma factor (stress)
    
    # Translation initiation
    'JCVISYN3A_0791': 'E',  # infA - IF-1
    'JCVISYN3A_0188': 'E',  # infB - IF-2
    'JCVISYN3A_0796': 'E',  # infC - IF-3
    
    # Translation elongation
    'JCVISYN3A_0094': 'E',  # tufA - EF-Tu
    'JCVISYN3A_0095': 'E',  # fusA - EF-G
    'JCVISYN3A_0797': 'E',  # tsf - EF-Ts
    
    # Translation termination
    'JCVISYN3A_0798': 'E',  # prfA - RF-1
    'JCVISYN3A_0799': 'E',  # prfB - RF-2
    
    # tRNA synthetases - ALL ESSENTIAL
    'JCVISYN3A_0476': 'E',  # alaS
    'JCVISYN3A_0838': 'E',  # argS
    'JCVISYN3A_0382': 'E',  # asnS
    'JCVISYN3A_0069': 'E',  # aspS
    'JCVISYN3A_0479': 'E',  # cysS
    'JCVISYN3A_0543': 'E',  # glnS
    'JCVISYN3A_0530': 'E',  # gltX (gluS)
    'JCVISYN3A_0070': 'E',  # glyS
    'JCVISYN3A_0542': 'E',  # hisS
    'JCVISYN3A_0523': 'E',  # ileS
    'JCVISYN3A_0482': 'E',  # leuS
    'JCVISYN3A_0250': 'E',  # lysS
    'JCVISYN3A_0221': 'E',  # metS
    'JCVISYN3A_0187': 'E',  # pheS
    'JCVISYN3A_0529': 'E',  # proS
    'JCVISYN3A_0687': 'E',  # serS
    'JCVISYN3A_0232': 'E',  # thrS
    'JCVISYN3A_0226': 'E',  # trpS
    'JCVISYN3A_0262': 'E',  # tyrS
    'JCVISYN3A_0375': 'E',  # valS
    
    # Ribosomal proteins (essential subset)
    'JCVISYN3A_0288': 'E',  # rpsA - 30S S1
    'JCVISYN3A_0795': 'E',  # rpsB - 30S S2
    'JCVISYN3A_0116': 'E',  # rpsC - 30S S3
    'JCVISYN3A_0117': 'E',  # rpsD - 30S S4
    'JCVISYN3A_0096': 'E',  # rplA - 50S L1
    'JCVISYN3A_0097': 'E',  # rplB - 50S L2
    'JCVISYN3A_0098': 'E',  # rplC - 50S L3
    'JCVISYN3A_0099': 'E',  # rplD - 50S L4
    
    # Cell division
    'JCVISYN3A_0516': 'E',  # ftsZ - division ring
    'JCVISYN3A_0517': 'E',  # ftsA - division
    'JCVISYN3A_0518': 'N',  # ftsW - division (non-essential in minimal)
    
    # Lipid synthesis
    'JCVISYN3A_0161': 'E',  # accA - acetyl-CoA carboxylase
    'JCVISYN3A_0162': 'E',  # accB - biotin carboxyl carrier
    'JCVISYN3A_0163': 'E',  # accC - biotin carboxylase
    'JCVISYN3A_0164': 'E',  # accD - carboxyltransferase
    'JCVISYN3A_0165': 'E',  # fabD - malonyl-CoA ACP transacylase
    'JCVISYN3A_0166': 'E',  # fabH - 3-oxoacyl-ACP synthase III
    'JCVISYN3A_0167': 'E',  # fabG - 3-oxoacyl-ACP reductase
    'JCVISYN3A_0168': 'E',  # fabF - 3-oxoacyl-ACP synthase II
    'JCVISYN3A_0169': 'E',  # fabA - 3-hydroxydecanoyl-ACP dehydratase
    'JCVISYN3A_0170': 'E',  # fabI - enoyl-ACP reductase
    'JCVISYN3A_0830': 'N',  # glpK - glycerol kinase
    'JCVISYN3A_0831': 'N',  # glpF - glycerol uptake facilitator
    
    # Transporters
    'JCVISYN3A_0549': 'N',  # fruK - fructose-specific
    'JCVISYN3A_0550': 'N',  # fruA - fructose PTS
    'JCVISYN3A_0684': 'N',  # ptsH - HPr
    'JCVISYN3A_0683': 'N',  # ptsI - enzyme I
    
    # Protein folding/quality
    'JCVISYN3A_0527': 'E',  # groEL - chaperonin
    'JCVISYN3A_0528': 'E',  # groES - co-chaperonin
    'JCVISYN3A_0524': 'Q',  # dnaK - Hsp70
    'JCVISYN3A_0525': 'Q',  # dnaJ - Hsp40
    'JCVISYN3A_0526': 'Q',  # grpE - nucleotide exchange
    'JCVISYN3A_0294': 'N',  # clpB - disaggregase
    'JCVISYN3A_0295': 'E',  # clpP - protease
    'JCVISYN3A_0296': 'E',  # clpX - unfoldase
    'JCVISYN3A_0297': 'E',  # lon - protease
    'JCVISYN3A_0298': 'E',  # ftsH - membrane protease
}

# Gene ID to name mapping
GENE_NAMES: Dict[str, Tuple[str, str]] = {
    'JCVISYN3A_0685': ('ptsG', 'Glucose PTS permease'),
    'JCVISYN3A_0233': ('pgi', 'Phosphoglucose isomerase'),
    'JCVISYN3A_0207': ('pfkA', 'Phosphofructokinase'),
    'JCVISYN3A_0352': ('fba', 'Fructose-bisphosphate aldolase'),
    'JCVISYN3A_0353': ('tpiA', 'Triose phosphate isomerase'),
    'JCVISYN3A_0314': ('gapA', 'Glyceraldehyde-3P dehydrogenase'),
    'JCVISYN3A_0315': ('pgk', 'Phosphoglycerate kinase'),
    'JCVISYN3A_0689': ('pgm', 'Phosphoglycerate mutase'),
    'JCVISYN3A_0231': ('eno', 'Enolase'),
    'JCVISYN3A_0546': ('pyk', 'Pyruvate kinase'),
    'JCVISYN3A_0449': ('ldh', 'Lactate dehydrogenase'),
    'JCVISYN3A_0589': ('pfl', 'Pyruvate formate lyase'),
    'JCVISYN3A_0484': ('pta', 'Phosphotransacetylase'),
    'JCVISYN3A_0485': ('ackA', 'Acetate kinase'),
    'JCVISYN3A_0439': ('zwf', 'Glucose-6P dehydrogenase'),
    'JCVISYN3A_0441': ('gnd', '6-Phosphogluconate dehydrogenase'),
    'JCVISYN3A_0509': ('rpe', 'Ribulose-5P epimerase'),
    'JCVISYN3A_0510': ('rpi', 'Ribose-5P isomerase'),
    'JCVISYN3A_0234': ('tkt', 'Transketolase'),
    'JCVISYN3A_0235': ('tal', 'Transaldolase'),
    'JCVISYN3A_0317': ('prsA', 'PRPP synthetase'),
    'JCVISYN3A_0416': ('ndk', 'Nucleoside diphosphate kinase'),
    'JCVISYN3A_0005': ('adk', 'Adenylate kinase'),
    'JCVISYN3A_0783': ('atpA', 'ATP synthase alpha'),
    'JCVISYN3A_0001': ('dnaA', 'Replication initiation'),
    'JCVISYN3A_0690': ('dnaE', 'DNA polymerase III'),
    'JCVISYN3A_0790': ('rpoA', 'RNA polymerase alpha'),
    'JCVISYN3A_0218': ('rpoB', 'RNA polymerase beta'),
    'JCVISYN3A_0094': ('tufA', 'Elongation factor Tu'),
    'JCVISYN3A_0095': ('fusA', 'Elongation factor G'),
    'JCVISYN3A_0516': ('ftsZ', 'Cell division protein'),
    'JCVISYN3A_0161': ('accA', 'Acetyl-CoA carboxylase'),
    'JCVISYN3A_0527': ('groEL', 'Chaperonin GroEL'),
    'JCVISYN3A_0528': ('groES', 'Co-chaperonin GroES'),
    'JCVISYN3A_0524': ('dnaK', 'Chaperone DnaK'),
    'JCVISYN3A_0476': ('alaS', 'Alanyl-tRNA synthetase'),
    'JCVISYN3A_0838': ('argS', 'Arginyl-tRNA synthetase'),
    'JCVISYN3A_0382': ('asnS', 'Asparaginyl-tRNA synthetase'),
    'JCVISYN3A_0069': ('aspS', 'Aspartyl-tRNA synthetase'),
    'JCVISYN3A_0479': ('cysS', 'Cysteinyl-tRNA synthetase'),
    'JCVISYN3A_0543': ('glnS', 'Glutaminyl-tRNA synthetase'),
    'JCVISYN3A_0530': ('gltX', 'Glutamyl-tRNA synthetase'),
    'JCVISYN3A_0070': ('glyS', 'Glycyl-tRNA synthetase'),
    'JCVISYN3A_0542': ('hisS', 'Histidyl-tRNA synthetase'),
    'JCVISYN3A_0523': ('ileS', 'Isoleucyl-tRNA synthetase'),
    'JCVISYN3A_0482': ('leuS', 'Leucyl-tRNA synthetase'),
    'JCVISYN3A_0250': ('lysS', 'Lysyl-tRNA synthetase'),
    'JCVISYN3A_0221': ('metS', 'Methionyl-tRNA synthetase'),
    'JCVISYN3A_0187': ('pheS', 'Phenylalanyl-tRNA synthetase'),
    'JCVISYN3A_0529': ('proS', 'Prolyl-tRNA synthetase'),
    'JCVISYN3A_0687': ('serS', 'Seryl-tRNA synthetase'),
    'JCVISYN3A_0232': ('thrS', 'Threonyl-tRNA synthetase'),
    'JCVISYN3A_0226': ('trpS', 'Tryptophanyl-tRNA synthetase'),
    'JCVISYN3A_0262': ('tyrS', 'Tyrosyl-tRNA synthetase'),
    'JCVISYN3A_0375': ('valS', 'Valyl-tRNA synthetase'),
}


def get_gene_name(gene_id: str) -> str:
    """Get short name for gene ID."""
    if gene_id in GENE_NAMES:
        return GENE_NAMES[gene_id][0]
    return gene_id


def get_gene_description(gene_id: str) -> str:
    """Get full description for gene ID."""
    if gene_id in GENE_NAMES:
        return GENE_NAMES[gene_id][1]
    return "Unknown"


def is_essential(gene_id: str) -> bool:
    """Check if gene is essential (E or Q in Hutchison data)."""
    label = GENE_ESSENTIALITY.get(gene_id, 'N')
    return label in ['E', 'Q']


def get_labeled_genes() -> List[str]:
    """Get list of genes with experimental essentiality labels."""
    return list(GENE_ESSENTIALITY.keys())


def get_essential_genes() -> List[str]:
    """Get list of essential genes (E or Q)."""
    return [g for g, label in GENE_ESSENTIALITY.items() if label in ['E', 'Q']]


def get_nonessential_genes() -> List[str]:
    """Get list of non-essential genes (N)."""
    return [g for g, label in GENE_ESSENTIALITY.items() if label == 'N']


def get_train_test_split(test_fraction: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Split genes into train/test sets for cross-validation.
    
    Stratified by essentiality to ensure both sets have similar class balance.
    """
    import random
    random.seed(seed)
    
    essential = get_essential_genes()
    nonessential = get_nonessential_genes()
    
    random.shuffle(essential)
    random.shuffle(nonessential)
    
    n_ess_test = int(len(essential) * test_fraction)
    n_noness_test = int(len(nonessential) * test_fraction)
    
    test_genes = essential[:n_ess_test] + nonessential[:n_noness_test]
    train_genes = essential[n_ess_test:] + nonessential[n_noness_test:]
    
    return train_genes, test_genes


# Summary statistics
def print_summary():
    """Print summary of essentiality data."""
    total = len(GENE_ESSENTIALITY)
    essential = len([g for g, l in GENE_ESSENTIALITY.items() if l == 'E'])
    quasi = len([g for g, l in GENE_ESSENTIALITY.items() if l == 'Q'])
    nonessential = len([g for g, l in GENE_ESSENTIALITY.items() if l == 'N'])
    
    print(f"Hutchison 2016 Essentiality Data:")
    print(f"  Total genes: {total}")
    print(f"  Essential (E): {essential} ({essential/total*100:.1f}%)")
    print(f"  Quasi-essential (Q): {quasi} ({quasi/total*100:.1f}%)")
    print(f"  Non-essential (N): {nonessential} ({nonessential/total*100:.1f}%)")
    print(f"  Positive class (E+Q): {essential+quasi} ({(essential+quasi)/total*100:.1f}%)")


if __name__ == "__main__":
    print_summary()
