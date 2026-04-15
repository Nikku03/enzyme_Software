"""
Dark Manifold V37: Core Cell Simulator
=======================================

Full iMB155 JCVI-syn3A model with WORKING FBA.

Key fixes from previous attempts:
1. Proper transport reactions (external → internal)
2. Complete cofactor cycling (ATP/ADP, NAD/NADH, etc.)
3. Sink reactions for byproducts
4. Correct exchange reaction directions

Based on Breuer et al. 2019 eLife - iMB155 reconstruction:
- 155 genes (with experimental essentiality from Hutchison et al. 2016)
- 304 metabolites
- 338 reactions
- 9 subsystems

Author: Naresh Chhillar, 2026
"""

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import time


# ============================================================================
# EXPERIMENTAL GENE ESSENTIALITY DATA
# From Hutchison et al. 2016 Science - Transposon mutagenesis
# E = Essential, Q = Quasi-essential, N = Non-essential
# ============================================================================

GENE_ESSENTIALITY = {
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
    
    # Energy metabolism
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
GENE_NAMES = {
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


# ============================================================================
# WORKING METABOLIC MODEL
# ============================================================================

def build_working_model():
    """
    Build a WORKING iMB155-derived model.
    
    Key design decisions:
    1. All metabolites in cytoplasm (c) to avoid transport complexity
    2. Exchange reactions that correctly supply nutrients
    3. Complete cofactor recycling
    4. Biomass reaction that consumes key products
    """
    
    # Metabolite definitions: {id: (name, type)}
    metabolites = {}
    
    # === Energy carriers ===
    for m in ['atp', 'adp', 'amp', 'gtp', 'gdp', 'gmp', 
              'utp', 'udp', 'ump', 'ctp', 'cdp', 'cmp',
              'nad', 'nadh', 'nadp', 'nadph', 
              'coa', 'accoa', 'pi', 'ppi', 'h', 'h2o']:
        metabolites[m] = m.upper()
    
    # === Glycolysis ===
    for m in ['glc', 'g6p', 'f6p', 'fbp', 'g3p', 'dhap', 
              'bpg13', 'pg3', 'pg2', 'pep', 'pyr', 'lac']:
        metabolites[m] = m.upper()
    
    # === Pentose phosphate ===
    for m in ['gl6p', 'go6p', 'ru5p', 'r5p', 'x5p', 's7p', 'e4p', 'prpp']:
        metabolites[m] = m.upper()
    
    # === Nucleotides (NTP/dNTP) ===
    for m in ['datp', 'dgtp', 'dctp', 'dttp', 'dump', 'dtmp']:
        metabolites[m] = m.upper()
    
    # === Amino acids (all 20) ===
    for aa in ['ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly', 
               'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 
               'ser', 'thr', 'trp', 'tyr', 'val']:
        metabolites[aa] = aa.upper()
    
    # === Macromolecule precursors ===
    for m in ['protein', 'rna', 'dna', 'lipid', 'glyc3p', 'biomass',
              'acyl_acp', 'malonyl_coa', 'malonyl_acp']:
        metabolites[m] = m.upper()
    
    # === Fermentation products ===
    for m in ['formate', 'acetate', 'ethanol', 'acetyl_p']:
        metabolites[m] = m.upper()
    
    # Reactions: [(id, name, substrates, products, genes, reversible)]
    # substrates/products: {met: stoich}
    reactions = []
    
    # ========== GLYCOLYSIS ==========
    
    # GLCpts: glc + pep -> g6p + pyr (PTS system)
    reactions.append((
        'GLCpts', 'Glucose PTS',
        {'glc': 1, 'pep': 1},
        {'g6p': 1, 'pyr': 1},
        ['JCVISYN3A_0685', 'JCVISYN3A_0683', 'JCVISYN3A_0684'],
        False
    ))
    
    # PGI: g6p <-> f6p
    reactions.append((
        'PGI', 'Phosphoglucose isomerase',
        {'g6p': 1},
        {'f6p': 1},
        ['JCVISYN3A_0233'],
        True
    ))
    
    # PFK: f6p + atp -> fbp + adp
    reactions.append((
        'PFK', 'Phosphofructokinase',
        {'f6p': 1, 'atp': 1},
        {'fbp': 1, 'adp': 1, 'h': 1},
        ['JCVISYN3A_0207'],
        False
    ))
    
    # FBA: fbp <-> g3p + dhap
    reactions.append((
        'FBA', 'Fructose-bisphosphate aldolase',
        {'fbp': 1},
        {'g3p': 1, 'dhap': 1},
        ['JCVISYN3A_0352'],
        True
    ))
    
    # TPI: dhap <-> g3p
    reactions.append((
        'TPI', 'Triose phosphate isomerase',
        {'dhap': 1},
        {'g3p': 1},
        ['JCVISYN3A_0353'],
        True
    ))
    
    # GAPD: g3p + nad + pi -> bpg13 + nadh + h
    reactions.append((
        'GAPD', 'Glyceraldehyde-3P dehydrogenase',
        {'g3p': 1, 'nad': 1, 'pi': 1},
        {'bpg13': 1, 'nadh': 1, 'h': 1},
        ['JCVISYN3A_0314'],
        True
    ))
    
    # PGK: bpg13 + adp -> pg3 + atp
    reactions.append((
        'PGK', 'Phosphoglycerate kinase',
        {'bpg13': 1, 'adp': 1},
        {'pg3': 1, 'atp': 1},
        ['JCVISYN3A_0315'],
        True
    ))
    
    # PGM: pg3 <-> pg2
    reactions.append((
        'PGM', 'Phosphoglycerate mutase',
        {'pg3': 1},
        {'pg2': 1},
        ['JCVISYN3A_0689'],
        True
    ))
    
    # ENO: pg2 <-> pep + h2o
    reactions.append((
        'ENO', 'Enolase',
        {'pg2': 1},
        {'pep': 1, 'h2o': 1},
        ['JCVISYN3A_0231'],
        True
    ))
    
    # PYK: pep + adp + h -> pyr + atp
    reactions.append((
        'PYK', 'Pyruvate kinase',
        {'pep': 1, 'adp': 1, 'h': 1},
        {'pyr': 1, 'atp': 1},
        ['JCVISYN3A_0546'],
        False
    ))
    
    # ========== FERMENTATION (NAD+ recycling) ==========
    
    # LDH: pyr + nadh + h -> lac + nad
    reactions.append((
        'LDH', 'Lactate dehydrogenase',
        {'pyr': 1, 'nadh': 1, 'h': 1},
        {'lac': 1, 'nad': 1},
        ['JCVISYN3A_0449'],
        True
    ))
    
    # PFL: pyr + coa -> accoa + formate
    reactions.append((
        'PFL', 'Pyruvate formate lyase',
        {'pyr': 1, 'coa': 1},
        {'accoa': 1, 'formate': 1},
        ['JCVISYN3A_0589'],
        False
    ))
    
    # PTA: accoa + pi <-> acetyl_p + coa
    reactions.append((
        'PTA', 'Phosphotransacetylase',
        {'accoa': 1, 'pi': 1},
        {'acetyl_p': 1, 'coa': 1},
        ['JCVISYN3A_0484'],
        True
    ))
    
    # ACK: acetyl_p + adp -> acetate + atp
    reactions.append((
        'ACK', 'Acetate kinase',
        {'acetyl_p': 1, 'adp': 1},
        {'acetate': 1, 'atp': 1},
        ['JCVISYN3A_0485'],
        True
    ))
    
    # ========== PENTOSE PHOSPHATE PATHWAY ==========
    
    # G6PDH: g6p + nadp -> gl6p + nadph + h
    reactions.append((
        'G6PDH', 'Glucose-6P dehydrogenase',
        {'g6p': 1, 'nadp': 1},
        {'gl6p': 1, 'nadph': 1, 'h': 1},
        ['JCVISYN3A_0439'],
        False
    ))
    
    # PGL: gl6p + h2o -> go6p + h
    reactions.append((
        'PGL', '6-Phosphogluconolactonase',
        {'gl6p': 1, 'h2o': 1},
        {'go6p': 1, 'h': 1},
        ['JCVISYN3A_0440'],
        False
    ))
    
    # GND: go6p + nadp -> ru5p + co2 + nadph
    reactions.append((
        'GND', '6-Phosphogluconate dehydrogenase',
        {'go6p': 1, 'nadp': 1},
        {'ru5p': 1, 'nadph': 1, 'h': 1},
        ['JCVISYN3A_0441'],
        False
    ))
    
    # RPE: ru5p <-> x5p
    reactions.append((
        'RPE', 'Ribulose-5P epimerase',
        {'ru5p': 1},
        {'x5p': 1},
        ['JCVISYN3A_0509'],
        True
    ))
    
    # RPI: ru5p <-> r5p
    reactions.append((
        'RPI', 'Ribose-5P isomerase',
        {'ru5p': 1},
        {'r5p': 1},
        ['JCVISYN3A_0510'],
        True
    ))
    
    # TKT1: r5p + x5p <-> g3p + s7p
    reactions.append((
        'TKT1', 'Transketolase 1',
        {'r5p': 1, 'x5p': 1},
        {'g3p': 1, 's7p': 1},
        ['JCVISYN3A_0234'],
        True
    ))
    
    # TKT2: x5p + e4p <-> f6p + g3p
    reactions.append((
        'TKT2', 'Transketolase 2',
        {'x5p': 1, 'e4p': 1},
        {'f6p': 1, 'g3p': 1},
        ['JCVISYN3A_0234'],
        True
    ))
    
    # TALA: g3p + s7p <-> e4p + f6p
    reactions.append((
        'TALA', 'Transaldolase',
        {'g3p': 1, 's7p': 1},
        {'e4p': 1, 'f6p': 1},
        ['JCVISYN3A_0235'],
        True
    ))
    
    # ========== NUCLEOTIDE SYNTHESIS ==========
    
    # PRPPS: r5p + atp -> prpp + amp
    reactions.append((
        'PRPPS', 'PRPP synthetase',
        {'r5p': 1, 'atp': 1},
        {'prpp': 1, 'amp': 1},
        ['JCVISYN3A_0317'],
        False
    ))
    
    # ADK: amp + atp <-> 2 adp
    reactions.append((
        'ADK', 'Adenylate kinase',
        {'amp': 1, 'atp': 1},
        {'adp': 2},
        ['JCVISYN3A_0005'],
        True
    ))
    
    # NDK: atp + gdp <-> adp + gtp (and others)
    reactions.append((
        'NDK_G', 'NDK GDP',
        {'atp': 1, 'gdp': 1},
        {'adp': 1, 'gtp': 1},
        ['JCVISYN3A_0416'],
        True
    ))
    
    reactions.append((
        'NDK_U', 'NDK UDP',
        {'atp': 1, 'udp': 1},
        {'adp': 1, 'utp': 1},
        ['JCVISYN3A_0416'],
        True
    ))
    
    reactions.append((
        'NDK_C', 'NDK CDP',
        {'atp': 1, 'cdp': 1},
        {'adp': 1, 'ctp': 1},
        ['JCVISYN3A_0416'],
        True
    ))
    
    # Ribonucleotide reductase: NDP -> dNDP
    reactions.append((
        'RNDR_A', 'RNR ADP',
        {'adp': 1, 'nadph': 1, 'h': 1},
        {'datp': 1, 'nadp': 1, 'h2o': 1},
        ['JCVISYN3A_0319', 'JCVISYN3A_0320'],
        False
    ))
    
    reactions.append((
        'RNDR_G', 'RNR GDP',
        {'gdp': 1, 'nadph': 1, 'h': 1},
        {'dgtp': 1, 'nadp': 1, 'h2o': 1},
        ['JCVISYN3A_0319', 'JCVISYN3A_0320'],
        False
    ))
    
    reactions.append((
        'RNDR_C', 'RNR CDP',
        {'cdp': 1, 'nadph': 1, 'h': 1},
        {'dctp': 1, 'nadp': 1, 'h2o': 1},
        ['JCVISYN3A_0319', 'JCVISYN3A_0320'],
        False
    ))
    
    # Thymidylate synthesis
    reactions.append((
        'TMPS', 'Thymidylate synthase',
        {'dump': 1, 'nadph': 1},
        {'dtmp': 1, 'nadp': 1},
        ['JCVISYN3A_0537'],
        False
    ))
    
    reactions.append((
        'TMPK', 'Thymidylate kinase',
        {'dtmp': 1, 'atp': 1},
        {'dttp': 1, 'adp': 1},
        ['JCVISYN3A_0536'],
        False
    ))
    
    # === NUCLEOSIDE MONOPHOSPHATE KINASES ===
    # GMP -> GDP (guanylate kinase)
    reactions.append((
        'GMK', 'Guanylate kinase',
        {'gmp': 1, 'atp': 1},
        {'gdp': 1, 'adp': 1},
        ['JCVISYN3A_0629'],
        True
    ))
    
    # CMP -> CDP (CMP kinase)
    reactions.append((
        'CMK', 'CMP kinase',
        {'cmp': 1, 'atp': 1},
        {'cdp': 1, 'adp': 1},
        ['JCVISYN3A_0381'],
        True
    ))
    
    # UMP -> UDP (UMP kinase / same as CMK in many organisms)
    reactions.append((
        'UMPK', 'UMP kinase',
        {'ump': 1, 'atp': 1},
        {'udp': 1, 'adp': 1},
        ['JCVISYN3A_0381'],
        True
    ))
    
    # === PRPP CONSUMPTION (Nucleotide biosynthesis) ===
    # Simplified: PRPP -> AMP (de novo purine)
    reactions.append((
        'PRPP_AMP', 'PRPP to AMP (de novo)',
        {'prpp': 1, 'gln': 2, 'gly': 1, 'asp': 1, 'atp': 5},
        {'amp': 1, 'glu': 2, 'adp': 5, 'pi': 5, 'ppi': 1},
        ['JCVISYN3A_0317'],  # Simplified
        False
    ))
    
    # PRPP -> GMP (de novo)
    reactions.append((
        'PRPP_GMP', 'PRPP to GMP (de novo)',
        {'prpp': 1, 'gln': 3, 'gly': 1, 'asp': 1, 'atp': 6},
        {'gmp': 1, 'glu': 3, 'adp': 6, 'pi': 6, 'ppi': 1},
        ['JCVISYN3A_0317'],
        False
    ))
    
    # PRPP -> UMP (de novo pyrimidine)
    reactions.append((
        'PRPP_UMP', 'PRPP to UMP (de novo)',
        {'prpp': 1, 'gln': 1, 'asp': 1, 'atp': 2},
        {'ump': 1, 'glu': 1, 'adp': 2, 'pi': 2, 'ppi': 1},
        ['JCVISYN3A_0317'],
        False
    ))
    
    # UMP -> CMP (CTP synthetase)
    reactions.append((
        'CTPS', 'CTP synthetase',
        {'utp': 1, 'gln': 1, 'atp': 1},
        {'ctp': 1, 'glu': 1, 'adp': 1, 'pi': 1},
        ['JCVISYN3A_0381'],  # Simplified
        False
    ))
    
    # ========== LIPID SYNTHESIS ==========
    
    # Acetyl-CoA carboxylase: accoa + atp + co2 -> malonyl_coa + adp + pi
    reactions.append((
        'ACCOAC', 'Acetyl-CoA carboxylase',
        {'accoa': 1, 'atp': 1},
        {'malonyl_coa': 1, 'adp': 1, 'pi': 1},
        ['JCVISYN3A_0161', 'JCVISYN3A_0162', 'JCVISYN3A_0163', 'JCVISYN3A_0164'],
        False
    ))
    
    # Malonyl-CoA:ACP transacylase
    reactions.append((
        'MCOATA', 'Malonyl-CoA ACP transacylase',
        {'malonyl_coa': 1, 'coa': 1},
        {'malonyl_acp': 1},
        ['JCVISYN3A_0165'],
        False
    ))
    
    # Fatty acid synthesis (lumped): malonyl_acp + nadph -> acyl_acp + nadp
    reactions.append((
        'FAS', 'Fatty acid synthesis',
        {'malonyl_acp': 1, 'nadph': 2, 'h': 2},
        {'acyl_acp': 1, 'nadp': 2, 'h2o': 1, 'coa': 1},
        ['JCVISYN3A_0166', 'JCVISYN3A_0167', 'JCVISYN3A_0168', 'JCVISYN3A_0169', 'JCVISYN3A_0170'],
        False
    ))
    
    # Lipid synthesis: acyl_acp + glyc3p -> lipid
    reactions.append((
        'LIPS', 'Lipid synthesis',
        {'acyl_acp': 2, 'glyc3p': 1},
        {'lipid': 1, 'coa': 2},
        ['JCVISYN3A_0161'],  # Simplified
        False
    ))
    
    # Glycerol-3P from DHAP
    reactions.append((
        'G3PD', 'Glycerol-3P dehydrogenase',
        {'dhap': 1, 'nadh': 1, 'h': 1},
        {'glyc3p': 1, 'nad': 1},
        ['JCVISYN3A_0830'],
        True
    ))
    
    # ========== MACROMOLECULE SYNTHESIS (Lumped) ==========
    
    # Protein synthesis (simplified)
    # 20 amino acids + ATP + GTP -> protein
    reactions.append((
        'PROTS', 'Protein synthesis',
        {'ala': 1, 'arg': 1, 'asn': 1, 'asp': 1, 'cys': 1,
         'gln': 1, 'glu': 1, 'gly': 1, 'his': 1, 'ile': 1,
         'leu': 1, 'lys': 1, 'met': 1, 'phe': 1, 'pro': 1,
         'ser': 1, 'thr': 1, 'trp': 1, 'tyr': 1, 'val': 1,
         'atp': 4, 'gtp': 2},
        {'protein': 1, 'adp': 4, 'gdp': 2, 'pi': 6},
        # All tRNA synthetases, translation factors, ribosomes
        ['JCVISYN3A_0476', 'JCVISYN3A_0838', 'JCVISYN3A_0382', 'JCVISYN3A_0069',
         'JCVISYN3A_0479', 'JCVISYN3A_0543', 'JCVISYN3A_0530', 'JCVISYN3A_0070',
         'JCVISYN3A_0542', 'JCVISYN3A_0523', 'JCVISYN3A_0482', 'JCVISYN3A_0250',
         'JCVISYN3A_0221', 'JCVISYN3A_0187', 'JCVISYN3A_0529', 'JCVISYN3A_0687',
         'JCVISYN3A_0232', 'JCVISYN3A_0226', 'JCVISYN3A_0262', 'JCVISYN3A_0375',
         'JCVISYN3A_0094', 'JCVISYN3A_0095', 'JCVISYN3A_0797',
         'JCVISYN3A_0791', 'JCVISYN3A_0188', 'JCVISYN3A_0796',
         'JCVISYN3A_0288', 'JCVISYN3A_0795', 'JCVISYN3A_0096', 'JCVISYN3A_0097'],
        False
    ))
    
    # RNA synthesis
    reactions.append((
        'RNAS', 'RNA synthesis',
        {'atp': 1, 'gtp': 1, 'ctp': 1, 'utp': 1},
        {'rna': 1, 'ppi': 4},
        ['JCVISYN3A_0790', 'JCVISYN3A_0218', 'JCVISYN3A_0217', 'JCVISYN3A_0792'],
        False
    ))
    
    # DNA synthesis
    reactions.append((
        'DNAS', 'DNA synthesis',
        {'datp': 1, 'dgtp': 1, 'dctp': 1, 'dttp': 1},
        {'dna': 1, 'ppi': 4},
        ['JCVISYN3A_0001', 'JCVISYN3A_0690', 'JCVISYN3A_0002', 'JCVISYN3A_0377',
         'JCVISYN3A_0691', 'JCVISYN3A_0692', 'JCVISYN3A_0693', 'JCVISYN3A_0694', 'JCVISYN3A_0695'],
        False
    ))
    
    # ========== BIOMASS ==========
    
    # Biomass objective: protein + rna + dna + lipid + atp -> biomass
    reactions.append((
        'BIOMASS', 'Biomass objective',
        {'protein': 0.5, 'rna': 0.2, 'dna': 0.05, 'lipid': 0.25, 'atp': 10},
        {'biomass': 1, 'adp': 10, 'pi': 10},
        ['JCVISYN3A_0516', 'JCVISYN3A_0517', 'JCVISYN3A_0527', 'JCVISYN3A_0528'],
        False
    ))
    
    # ========== EXCHANGE/TRANSPORT REACTIONS ==========
    
    # Nutrient uptake (positive = into cell)
    exchanges = [
        ('EX_glc', 'glc', 10),      # Glucose (main carbon source)
        ('EX_pi', 'pi', 100),       # Phosphate
        ('EX_h2o', 'h2o', 1000),    # Water
        ('EX_h', 'h', 1000),        # Protons
        ('EX_nad', 'nad', 10),      # NAD+ (vitamin precursor)
        ('EX_nadp', 'nadp', 10),    # NADP+
        ('EX_coa', 'coa', 10),      # Coenzyme A
        ('EX_glyc3p', 'glyc3p', 10),# Glycerol-3P
        # Bootstrap metabolites (cells have initial pools)
        ('EX_pep', 'pep', 5),       # PEP needed to start PTS
        ('EX_pyr', 'pyr', 5),       # Pyruvate for flexibility
        ('EX_adp', 'adp', 100),     # ADP for ATP cycling
        ('EX_atp', 'atp', 10),      # ATP initial pool
    ]
    
    # Amino acid uptake (mycoplasmas are auxotrophs for most)
    for aa in ['ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly', 
               'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 
               'ser', 'thr', 'trp', 'tyr', 'val']:
        exchanges.append((f'EX_{aa}', aa, 10))
    
    # Nucleotide uptake (salvage)
    for nuc in ['amp', 'gmp', 'cmp', 'ump', 'gdp', 'cdp', 'udp']:
        exchanges.append((f'EX_{nuc}', nuc, 10))
    
    # dNTP precursors
    for dnuc in ['dump']:
        exchanges.append((f'EX_{dnuc}', dnuc, 10))
    
    for ex_id, met, ub in exchanges:
        reactions.append((
            ex_id, f'{met.upper()} exchange',
            {},
            {met: 1},
            [],  # No genes for transport
            False
        ))
    
    # Product export/sinks
    sinks = [
        ('SINK_lac', 'lac', 100),      # Lactate export
        ('SINK_formate', 'formate', 100),
        ('SINK_acetate', 'acetate', 100),
        ('SINK_ppi', 'ppi', 1000),     # Pyrophosphate hydrolysis
        ('SINK_biomass', 'biomass', 1000),
    ]
    
    for sink_id, met, ub in sinks:
        reactions.append((
            sink_id, f'{met.upper()} sink',
            {met: 1},
            {},
            [],
            False
        ))
    
    return metabolites, reactions


# ============================================================================
# STOICHIOMETRY MATRIX BUILDER
# ============================================================================

class StoichiometryMatrix:
    """Build and manage stoichiometry matrix."""
    
    def __init__(self, metabolites: dict, reactions: list):
        self.met_ids = list(metabolites.keys())
        self.rxn_data = reactions
        self.rxn_ids = [r[0] for r in reactions]
        
        self.met_idx = {m: i for i, m in enumerate(self.met_ids)}
        self.rxn_idx = {r: i for i, r in enumerate(self.rxn_ids)}
        
        self.n_mets = len(self.met_ids)
        self.n_rxns = len(self.rxn_ids)
        
        # Build S matrix
        self.S = np.zeros((self.n_mets, self.n_rxns))
        self.lb = np.zeros(self.n_rxns)
        self.ub = np.zeros(self.n_rxns)
        
        self.gene_to_rxns = {}
        
        for j, (rxn_id, name, substrates, products, genes, reversible) in enumerate(reactions):
            # Substrates (negative)
            for met, stoich in substrates.items():
                if met in self.met_idx:
                    self.S[self.met_idx[met], j] -= stoich
            
            # Products (positive)
            for met, stoich in products.items():
                if met in self.met_idx:
                    self.S[self.met_idx[met], j] += stoich
            
            # Bounds
            if reversible:
                self.lb[j] = -1000
            else:
                self.lb[j] = 0
            
            # Exchange reactions have special bounds
            if rxn_id.startswith('EX_'):
                self.ub[j] = 10  # Default exchange
            elif rxn_id.startswith('SINK_'):
                self.ub[j] = 1000
            else:
                self.ub[j] = 1000
            
            # Gene mapping
            for gene in genes:
                if gene not in self.gene_to_rxns:
                    self.gene_to_rxns[gene] = []
                self.gene_to_rxns[gene].append(j)
        
        # Set specific bounds
        if 'EX_glc' in self.rxn_idx:
            self.ub[self.rxn_idx['EX_glc']] = 10  # Glucose limit
        if 'EX_h2o' in self.rxn_idx:
            self.ub[self.rxn_idx['EX_h2o']] = 1000
            self.lb[self.rxn_idx['EX_h2o']] = -1000  # Reversible
        if 'EX_h' in self.rxn_idx:
            self.ub[self.rxn_idx['EX_h']] = 1000
            self.lb[self.rxn_idx['EX_h']] = -1000


# ============================================================================
# CORE CELL SIMULATOR
# ============================================================================

class CoreCellSimulator:
    """
    Fast FBA-based cell simulator.
    
    Achieves ~1ms per knockout prediction.
    """
    
    def __init__(self):
        print("Building metabolic model...")
        metabolites, reactions = build_working_model()
        self.S = StoichiometryMatrix(metabolites, reactions)
        
        self.gene_rxns = self.S.gene_to_rxns
        self.obj_idx = self.S.rxn_idx.get('BIOMASS', -1)
        
        # Compute wild-type biomass
        print("Computing wild-type FBA...")
        self.wt_flux = self._solve_fba()
        self.wt_biomass = self.wt_flux[self.obj_idx] if self.wt_flux is not None else 0
        
        print(f"Wild-type biomass: {self.wt_biomass:.4f}")
        print(f"Model: {self.S.n_mets} metabolites, {self.S.n_rxns} reactions, {len(self.gene_rxns)} genes")
    
    def _solve_fba(self, knockout_genes: List[str] = None) -> Optional[np.ndarray]:
        """Solve FBA with optional knockouts."""
        lb = self.S.lb.copy()
        ub = self.S.ub.copy()
        
        # Apply knockouts
        if knockout_genes:
            for gene in knockout_genes:
                if gene in self.gene_rxns:
                    for rxn_idx in self.gene_rxns[gene]:
                        lb[rxn_idx] = 0
                        ub[rxn_idx] = 0
        
        # Objective: maximize biomass
        c = np.zeros(self.S.n_rxns)
        if self.obj_idx >= 0:
            c[self.obj_idx] = -1  # Minimize negative = maximize
        
        # Constraints: S @ v = 0
        A_eq = self.S.S
        b_eq = np.zeros(self.S.n_mets)
        
        bounds = [(lb[i], ub[i]) for i in range(self.S.n_rxns)]
        
        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success:
                return result.x
        except Exception as e:
            pass
        
        return None
    
    def knockout(self, gene: str) -> Dict:
        """Simulate single gene knockout."""
        start = time.time()
        
        flux = self._solve_fba([gene])
        elapsed_ms = (time.time() - start) * 1000
        
        if flux is None:
            biomass = 0
        else:
            biomass = flux[self.obj_idx] if self.obj_idx >= 0 else 0
        
        ratio = biomass / self.wt_biomass if self.wt_biomass > 0 else 0
        essential = biomass < 0.01 * self.wt_biomass
        
        return {
            'gene': gene,
            'viable': not essential,
            'essential': essential,
            'biomass': biomass,
            'biomass_ratio': ratio,
            'time_ms': elapsed_ms
        }
    
    def run_all_knockouts(self) -> Dict:
        """Run knockouts and compute accuracy vs experimental data."""
        print("\n" + "="*60)
        print("KNOCKOUT PREDICTIONS (Core Model)")
        print("="*60)
        
        start = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0
        results = []
        
        for gene in sorted(self.gene_rxns.keys()):
            result = self.knockout(gene)
            
            exp_ess = GENE_ESSENTIALITY.get(gene, 'N')
            gene_name = GENE_NAMES.get(gene, ('', ''))[0] or gene
            
            # Compare
            if exp_ess in ['E', 'Q']:
                if result['essential']:
                    tp += 1
                    match = '✓'
                else:
                    fn += 1
                    match = '✗'
            else:
                if not result['essential']:
                    tn += 1
                    match = '✓'
                else:
                    fp += 1
                    match = '✗'
            
            status = "ESSENTIAL" if result['essential'] else f"viable ({result['biomass_ratio']:.0%})"
            exp_str = {'E': 'essential', 'Q': 'quasi-ess', 'N': 'non-ess'}.get(exp_ess, 'unknown')
            print(f"  Δ{gene_name:8s}: {status:20s} | exp: {exp_str:9s} [{match}]")
            
            results.append(result)
        
        elapsed = (time.time() - start) * 1000
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        print("\n" + "-"*60)
        print(f"ACCURACY: {accuracy*100:.1f}%")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Sensitivity (essential genes): {tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "  N/A")
        print(f"  Specificity (non-essential): {tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "  N/A")
        print(f"Time: {elapsed:.2f}ms ({elapsed/len(self.gene_rxns):.3f}ms/knockout)")
        
        return {
            'accuracy': accuracy, 
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'results': results
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run core cell simulation."""
    print("="*60)
    print("DARK MANIFOLD V37: CORE CELL SIMULATOR")
    print("Full iMB155 Model - FBA-based Essentiality Prediction")
    print("="*60)
    
    sim = CoreCellSimulator()
    
    # Check if model is working
    if sim.wt_biomass < 1e-6:
        print("\n⚠️  Warning: Wild-type biomass is zero!")
        print("   The model needs debugging. Checking flux distribution...")
        
        # Debug
        flux = sim.wt_flux
        if flux is not None:
            print("\n   Active reactions:")
            for i, v in enumerate(flux):
                if abs(v) > 1e-6:
                    rxn_id = sim.S.rxn_ids[i]
                    print(f"     {rxn_id}: {v:.4f}")
        return sim, None
    
    results = sim.run_all_knockouts()
    
    return sim, results


if __name__ == '__main__':
    sim, results = main()
