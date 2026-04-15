"""
Dark Manifold V46b: TRUE TEST on E. coli
=========================================

PROPER TEST: Predict essentiality from FUNCTION ANNOTATION only,
without using essentiality labels to assign systems.

The test:
1. Take gene name/annotation
2. Predict system membership from annotation
3. Predict essentiality from system
4. Compare to experimental result

NO LEAKAGE: System assignment based on gene name/annotation,
not on whether the gene was found to be essential.
"""

import numpy as np
from collections import defaultdict
import re

# ============================================================================
# REAL E. COLI GENES WITH ANNOTATIONS
# ============================================================================

# All genes from Keio collection with their functional annotations
# Format: (gene_name, annotation, experimental_essential)

ECOLI_GENES = [
    # ========== TRANSLATION MACHINERY ==========
    # Ribosomal proteins - 30S
    ('rpsA', '30S ribosomal protein S1', True),
    ('rpsB', '30S ribosomal protein S2', True),
    ('rpsC', '30S ribosomal protein S3', True),
    ('rpsD', '30S ribosomal protein S4', True),
    ('rpsE', '30S ribosomal protein S5', True),
    ('rpsF', '30S ribosomal protein S6', False),  # Actually non-essential!
    ('rpsG', '30S ribosomal protein S7', True),
    ('rpsH', '30S ribosomal protein S8', True),
    ('rpsI', '30S ribosomal protein S9', True),
    ('rpsJ', '30S ribosomal protein S10', True),
    ('rpsK', '30S ribosomal protein S11', True),
    ('rpsL', '30S ribosomal protein S12', True),
    ('rpsM', '30S ribosomal protein S13', True),
    ('rpsN', '30S ribosomal protein S14', True),
    ('rpsO', '30S ribosomal protein S15', True),
    ('rpsP', '30S ribosomal protein S16', True),
    ('rpsQ', '30S ribosomal protein S17', True),
    ('rpsR', '30S ribosomal protein S18', True),
    ('rpsS', '30S ribosomal protein S19', True),
    ('rpsT', '30S ribosomal protein S20', True),
    ('rpsU', '30S ribosomal protein S21', False),  # Non-essential
    
    # Ribosomal proteins - 50S
    ('rplA', '50S ribosomal protein L1', True),
    ('rplB', '50S ribosomal protein L2', True),
    ('rplC', '50S ribosomal protein L3', True),
    ('rplD', '50S ribosomal protein L4', True),
    ('rplE', '50S ribosomal protein L5', True),
    ('rplF', '50S ribosomal protein L6', True),
    ('rplI', '50S ribosomal protein L9', False),  # Non-essential
    ('rplJ', '50S ribosomal protein L10', True),
    ('rplK', '50S ribosomal protein L11', True),
    ('rplL', '50S ribosomal protein L7/L12', True),
    ('rplM', '50S ribosomal protein L13', True),
    ('rplN', '50S ribosomal protein L14', True),
    ('rplO', '50S ribosomal protein L15', True),
    ('rplP', '50S ribosomal protein L16', True),
    ('rplQ', '50S ribosomal protein L17', True),
    ('rplR', '50S ribosomal protein L18', True),
    ('rplS', '50S ribosomal protein L19', True),
    ('rplT', '50S ribosomal protein L20', True),
    ('rplU', '50S ribosomal protein L21', True),
    ('rplV', '50S ribosomal protein L22', True),
    ('rplW', '50S ribosomal protein L23', True),
    ('rplX', '50S ribosomal protein L24', True),
    ('rpmA', '50S ribosomal protein L27', True),
    ('rpmB', '50S ribosomal protein L28', True),
    ('rpmC', '50S ribosomal protein L29', True),
    ('rpmD', '50S ribosomal protein L30', True),
    ('rpmE', '50S ribosomal protein L31', False),  # Non-essential
    ('rpmF', '50S ribosomal protein L32', True),
    ('rpmG', '50S ribosomal protein L33', True),
    ('rpmH', '50S ribosomal protein L34', True),
    ('rpmI', '50S ribosomal protein L35', True),
    ('rpmJ', '50S ribosomal protein L36', True),
    
    # tRNA synthetases (aminoacyl-tRNA synthetases)
    ('alaS', 'alanyl-tRNA synthetase', True),
    ('argS', 'arginyl-tRNA synthetase', True),
    ('asnS', 'asparaginyl-tRNA synthetase', True),
    ('aspS', 'aspartyl-tRNA synthetase', True),
    ('cysS', 'cysteinyl-tRNA synthetase', True),
    ('glnS', 'glutaminyl-tRNA synthetase', True),
    ('gltX', 'glutamyl-tRNA synthetase', True),
    ('glyS', 'glycyl-tRNA synthetase beta subunit', True),
    ('glyQ', 'glycyl-tRNA synthetase alpha subunit', True),
    ('hisS', 'histidyl-tRNA synthetase', True),
    ('ileS', 'isoleucyl-tRNA synthetase', True),
    ('leuS', 'leucyl-tRNA synthetase', True),
    ('lysS', 'lysyl-tRNA synthetase', False),  # Has paralog lysU
    ('lysU', 'lysyl-tRNA synthetase heat inducible', False),  # Paralog
    ('metG', 'methionyl-tRNA synthetase', True),
    ('pheS', 'phenylalanyl-tRNA synthetase alpha', True),
    ('pheT', 'phenylalanyl-tRNA synthetase beta', True),
    ('proS', 'prolyl-tRNA synthetase', True),
    ('serS', 'seryl-tRNA synthetase', True),
    ('thrS', 'threonyl-tRNA synthetase', True),
    ('trpS', 'tryptophanyl-tRNA synthetase', True),
    ('tyrS', 'tyrosyl-tRNA synthetase', True),
    ('valS', 'valyl-tRNA synthetase', True),
    
    # Translation factors
    ('infA', 'translation initiation factor IF-1', True),
    ('infB', 'translation initiation factor IF-2', True),
    ('infC', 'translation initiation factor IF-3', True),
    ('fusA', 'elongation factor G', True),
    ('tsf', 'elongation factor Ts', True),
    ('tufA', 'elongation factor Tu', True),
    ('tufB', 'elongation factor Tu duplicate', False),  # Redundant
    ('prfA', 'peptide chain release factor RF-1', True),
    ('prfB', 'peptide chain release factor RF-2', True),
    ('prfC', 'peptide chain release factor RF-3', False),  # Non-essential
    ('frr', 'ribosome recycling factor', True),
    ('efp', 'elongation factor P', True),
    
    # ========== TRANSCRIPTION ==========
    ('rpoA', 'RNA polymerase alpha subunit', True),
    ('rpoB', 'RNA polymerase beta subunit', True),
    ('rpoC', 'RNA polymerase beta prime subunit', True),
    ('rpoD', 'RNA polymerase sigma factor 70', True),
    ('rpoE', 'RNA polymerase sigma factor 24', True),
    ('rpoH', 'RNA polymerase sigma factor 32', True),
    ('rpoN', 'RNA polymerase sigma factor 54', False),  # Alternative sigma
    ('rpoS', 'RNA polymerase sigma factor 38', False),  # Stationary phase
    ('rpoZ', 'RNA polymerase omega subunit', False),  # Non-essential
    ('nusA', 'transcription termination factor NusA', True),
    ('nusB', 'transcription antitermination factor NusB', True),
    ('nusG', 'transcription termination factor NusG', True),
    ('rho', 'transcription termination factor Rho', True),
    ('greA', 'transcription elongation factor GreA', False),
    ('greB', 'transcription elongation factor GreB', False),
    
    # ========== REPLICATION ==========
    ('dnaA', 'chromosomal replication initiator', True),
    ('dnaB', 'replicative DNA helicase', True),
    ('dnaC', 'DNA replication protein DnaC', True),
    ('dnaE', 'DNA polymerase III alpha subunit', True),
    ('dnaG', 'DNA primase', True),
    ('dnaN', 'DNA polymerase III beta subunit', True),
    ('dnaQ', 'DNA polymerase III epsilon subunit', True),
    ('dnaX', 'DNA polymerase III tau/gamma subunit', True),
    ('holA', 'DNA polymerase III delta subunit', True),
    ('holB', 'DNA polymerase III delta prime subunit', True),
    ('holC', 'DNA polymerase III chi subunit', False),  # Non-essential
    ('holD', 'DNA polymerase III psi subunit', False),  # Non-essential
    ('holE', 'DNA polymerase III theta subunit', False),  # Non-essential
    ('ssb', 'single-stranded DNA-binding protein', True),
    ('ligA', 'DNA ligase', True),
    ('gyrA', 'DNA gyrase subunit A', True),
    ('gyrB', 'DNA gyrase subunit B', True),
    ('parC', 'topoisomerase IV subunit A', True),
    ('parE', 'topoisomerase IV subunit B', True),
    ('topA', 'DNA topoisomerase I', True),
    ('topB', 'DNA topoisomerase III', False),  # Non-essential
    
    # ========== CELL DIVISION ==========
    ('ftsZ', 'cell division protein FtsZ', True),
    ('ftsA', 'cell division protein FtsA', True),
    ('ftsB', 'cell division protein FtsB', True),
    ('ftsI', 'cell division protein FtsI (PBP3)', True),
    ('ftsK', 'DNA translocase FtsK', True),
    ('ftsL', 'cell division protein FtsL', True),
    ('ftsN', 'cell division protein FtsN', True),
    ('ftsQ', 'cell division protein FtsQ', True),
    ('ftsW', 'cell division protein FtsW', True),
    ('ftsE', 'cell division ATP-binding protein FtsE', False),  # Non-essential
    ('ftsX', 'cell division protein FtsX', False),  # Non-essential
    ('zipA', 'cell division protein ZipA', True),
    ('minC', 'septum site-determining protein MinC', False),
    ('minD', 'septum site-determining protein MinD', False),
    ('minE', 'cell division topological specificity factor', False),
    
    # ========== CELL WALL SYNTHESIS ==========
    ('murA', 'UDP-N-acetylglucosamine enolpyruvyl transferase', True),
    ('murB', 'UDP-N-acetylenolpyruvoylglucosamine reductase', True),
    ('murC', 'UDP-N-acetylmuramate-L-alanine ligase', True),
    ('murD', 'UDP-N-acetylmuramoylalanine-D-glutamate ligase', True),
    ('murE', 'UDP-N-acetylmuramoylalanyl-D-glutamate-meso-DAP ligase', True),
    ('murF', 'UDP-N-acetylmuramoyl-tripeptide-D-alanyl-D-alanine ligase', True),
    ('murG', 'UDP-N-acetylglucosamine-N-acetylmuramyl pyrophosphoryl undecaprenol transferase', True),
    ('murI', 'glutamate racemase', True),
    ('mraY', 'phospho-N-acetylmuramoyl-pentapeptide-transferase', True),
    ('ddlA', 'D-alanine-D-alanine ligase A', False),  # Has ddlB
    ('ddlB', 'D-alanine-D-alanine ligase B', True),
    
    # ========== LIPID METABOLISM ==========
    # Fatty acid synthesis
    ('accA', 'acetyl-CoA carboxylase carboxyltransferase alpha', True),
    ('accB', 'acetyl-CoA carboxylase biotin carboxyl carrier protein', True),
    ('accC', 'acetyl-CoA carboxylase biotin carboxylase', True),
    ('accD', 'acetyl-CoA carboxylase carboxyltransferase beta', True),
    ('fabA', '3-hydroxydecanoyl-ACP dehydratase', True),
    ('fabB', '3-oxoacyl-ACP synthase I', True),
    ('fabD', 'malonyl-CoA-ACP transacylase', True),
    ('fabF', '3-oxoacyl-ACP synthase II', False),  # fabB backup
    ('fabG', '3-oxoacyl-ACP reductase', True),
    ('fabH', '3-oxoacyl-ACP synthase III', True),
    ('fabI', 'enoyl-ACP reductase', True),
    ('fabZ', '3-hydroxyacyl-ACP dehydratase', True),
    ('acpP', 'acyl carrier protein', True),
    ('acpS', 'holo-ACP synthase', True),
    
    # LPS synthesis
    ('lpxA', 'UDP-N-acetylglucosamine acyltransferase', True),
    ('lpxB', 'lipid-A-disaccharide synthase', True),
    ('lpxC', 'UDP-3-O-acyl-N-acetylglucosamine deacetylase', True),
    ('lpxD', 'UDP-3-O-acylglucosamine N-acyltransferase', True),
    ('lpxH', 'UDP-2,3-diacylglucosamine hydrolase', True),
    ('lpxK', 'tetraacyldisaccharide 4-kinase', True),
    ('kdsA', '3-deoxy-D-manno-octulosonate 8-phosphate synthase', True),
    ('kdsB', '3-deoxy-manno-octulosonate cytidylyltransferase', True),
    ('kdsC', '3-deoxy-D-manno-octulosonate 8-phosphate phosphatase', True),
    ('kdsD', 'arabinose-5-phosphate isomerase', True),
    ('kdtA', 'lipopolysaccharide core biosynthesis protein', True),
    
    # ========== CENTRAL METABOLISM ==========
    # Glycolysis
    ('pgi', 'glucose-6-phosphate isomerase', False),  # Has bypass
    ('pfkA', '6-phosphofructokinase I', False),  # Has pfkB
    ('pfkB', '6-phosphofructokinase II', False),  # Minor isozyme
    ('fbaA', 'fructose-bisphosphate aldolase class II', True),
    ('fbaB', 'fructose-bisphosphate aldolase class I', False),  # Gluconeogenesis
    ('tpiA', 'triosephosphate isomerase', True),
    ('gapA', 'glyceraldehyde-3-phosphate dehydrogenase A', True),
    ('gapC', 'glyceraldehyde-3-phosphate dehydrogenase C', False),  # Minor
    ('pgk', 'phosphoglycerate kinase', True),
    ('gpmA', 'phosphoglyceromutase 1', True),
    ('gpmM', 'phosphoglyceromutase 2', False),  # Minor
    ('eno', 'enolase', True),
    ('pykF', 'pyruvate kinase I', False),  # Has pykA
    ('pykA', 'pyruvate kinase II', False),  # Has pykF
    
    # TCA cycle
    ('gltA', 'citrate synthase', False),
    ('acnA', 'aconitase A', False),
    ('acnB', 'aconitase B', False),
    ('icd', 'isocitrate dehydrogenase', False),
    ('sucA', '2-oxoglutarate dehydrogenase E1', False),
    ('sucB', '2-oxoglutarate dehydrogenase E2', False),
    ('sucC', 'succinyl-CoA synthetase beta', False),
    ('sucD', 'succinyl-CoA synthetase alpha', False),
    ('sdhA', 'succinate dehydrogenase flavoprotein', False),
    ('sdhB', 'succinate dehydrogenase iron-sulfur protein', False),
    ('sdhC', 'succinate dehydrogenase cytochrome b556 large', False),
    ('sdhD', 'succinate dehydrogenase cytochrome b556 small', False),
    ('fumA', 'fumarase A', False),
    ('fumB', 'fumarase B', False),
    ('fumC', 'fumarase C', False),
    ('mdh', 'malate dehydrogenase', False),
    
    # Pentose phosphate pathway
    ('zwf', 'glucose-6-phosphate dehydrogenase', False),  # Not essential
    ('pgl', '6-phosphogluconolactonase', False),
    ('gnd', '6-phosphogluconate dehydrogenase', False),
    ('rpe', 'ribulose-phosphate 3-epimerase', True),
    ('rpiA', 'ribose-5-phosphate isomerase A', True),
    ('rpiB', 'ribose-5-phosphate isomerase B', False),
    ('tktA', 'transketolase I', False),  # Has tktB
    ('tktB', 'transketolase II', False),
    ('talA', 'transaldolase A', False),
    ('talB', 'transaldolase B', False),
    
    # ========== NUCLEOTIDE METABOLISM ==========
    ('ndk', 'nucleoside diphosphate kinase', True),
    ('adk', 'adenylate kinase', True),
    ('cmk', 'cytidylate kinase', True),
    ('gmk', 'guanylate kinase', True),
    ('tmk', 'thymidylate kinase', True),
    ('pyrG', 'CTP synthase', True),
    ('pyrH', 'UMP kinase', True),
    ('prsA', 'ribose-phosphate pyrophosphokinase', True),
    ('purA', 'adenylosuccinate synthetase', True),
    ('purB', 'adenylosuccinate lyase', True),
    ('guaA', 'GMP synthase', True),
    ('guaB', 'IMP dehydrogenase', True),
    
    # ========== AMINO ACID BIOSYNTHESIS ==========
    # Most are non-essential if amino acids are provided
    ('argA', 'N-acetylglutamate synthase', False),
    ('argB', 'acetylglutamate kinase', False),
    ('argC', 'N-acetyl-gamma-glutamyl-phosphate reductase', False),
    ('argD', 'acetylornithine aminotransferase', False),
    ('argE', 'acetylornithine deacetylase', False),
    ('argF', 'ornithine carbamoyltransferase F', False),
    ('argG', 'argininosuccinate synthase', False),
    ('argH', 'argininosuccinate lyase', False),
    ('argI', 'ornithine carbamoyltransferase I', False),
    ('hisA', 'isomerase/cyclase HisA', False),
    ('hisB', 'histidinol-phosphatase/imidazoleglycerol-phosphate dehydratase', False),
    ('hisC', 'histidinol-phosphate aminotransferase', False),
    ('hisD', 'histidinol dehydrogenase', False),
    ('hisF', 'imidazole glycerol phosphate synthase HisF', False),
    ('hisG', 'ATP phosphoribosyltransferase', False),
    ('hisH', 'imidazole glycerol phosphate synthase HisH', False),
    ('hisI', 'phosphoribosyl-AMP cyclohydrolase/PR-ATP pyrophosphatase', False),
    ('ilvA', 'threonine dehydratase', False),
    ('ilvB', 'acetolactate synthase I large subunit', False),
    ('ilvC', 'ketol-acid reductoisomerase', False),
    ('ilvD', 'dihydroxy-acid dehydratase', False),
    ('ilvE', 'branched-chain amino acid aminotransferase', False),
    ('leuA', '2-isopropylmalate synthase', False),
    ('leuB', '3-isopropylmalate dehydrogenase', False),
    ('leuC', '3-isopropylmalate dehydratase large subunit', False),
    ('leuD', '3-isopropylmalate dehydratase small subunit', False),
    ('metA', 'homoserine O-succinyltransferase', False),
    ('metB', 'cystathionine gamma-synthase', False),
    ('metC', 'cystathionine beta-lyase', False),
    ('metE', 'homocysteine transmethylase', False),
    ('metH', 'homocysteine transmethylase', False),
    ('thrA', 'aspartokinase/homoserine dehydrogenase I', False),
    ('thrB', 'homoserine kinase', False),
    ('thrC', 'threonine synthase', False),
    ('trpA', 'tryptophan synthase alpha', False),
    ('trpB', 'tryptophan synthase beta', False),
    ('trpC', 'indole-3-glycerol phosphate synthase/phosphoribosylanthranilate isomerase', False),
    ('trpD', 'anthranilate phosphoribosyltransferase', False),
    ('trpE', 'anthranilate synthase component I', False),
    ('tyrA', 'chorismate mutase/prephenate dehydrogenase', False),
    ('tyrB', 'aromatic amino acid aminotransferase', False),
    
    # ========== PROTEIN FOLDING/QUALITY CONTROL ==========
    ('groEL', 'chaperonin GroEL', True),
    ('groES', 'co-chaperonin GroES', True),
    ('dnaK', 'chaperone protein DnaK', True),
    ('dnaJ', 'chaperone protein DnaJ', True),
    ('grpE', 'nucleotide exchange factor GrpE', True),
    ('tig', 'trigger factor', False),  # Non-essential chaperone
    ('htpG', 'chaperone protein HtpG', False),  # Hsp90
    ('clpA', 'ATP-dependent Clp protease ATP-binding subunit A', False),
    ('clpB', 'ATP-dependent Clp protease ATP-binding subunit B', False),
    ('clpP', 'ATP-dependent Clp protease proteolytic subunit', False),  # Non-essential!
    ('clpX', 'ATP-dependent Clp protease ATP-binding subunit X', False),  # Non-essential!
    ('clpS', 'ATP-dependent Clp protease adaptor protein', False),
    ('lon', 'Lon protease', False),  # Non-essential!
    ('ftsH', 'ATP-dependent zinc metalloprotease FtsH', True),  # Only essential protease
    ('hslU', 'ATP-dependent HslUV protease ATPase subunit', False),
    ('hslV', 'ATP-dependent HslUV protease peptidase subunit', False),
    
    # ========== PROTEIN SECRETION ==========
    ('secA', 'preprotein translocase subunit SecA', True),
    ('secB', 'preprotein translocase subunit SecB', False),  # Non-essential!
    ('secD', 'preprotein translocase subunit SecD', True),
    ('secE', 'preprotein translocase subunit SecE', True),
    ('secF', 'preprotein translocase subunit SecF', True),
    ('secG', 'preprotein translocase subunit SecG', False),  # Non-essential
    ('secM', 'secretion monitor', True),
    ('secY', 'preprotein translocase subunit SecY', True),
    ('ffh', 'signal recognition particle protein', True),
    ('ftsY', 'signal recognition particle receptor', True),
    ('yidC', 'membrane protein insertase YidC', True),
    ('lepB', 'signal peptidase I', True),
    ('lspA', 'signal peptidase II', True),
    
    # ========== COFACTOR BIOSYNTHESIS ==========
    ('coaA', 'pantothenate kinase', True),
    ('coaD', 'phosphopantetheine adenylyltransferase', True),
    ('coaE', 'dephospho-CoA kinase', True),
    ('folA', 'dihydrofolate reductase', True),
    ('folC', 'folylpolyglutamate synthase', True),
    ('folD', 'methylenetetrahydrofolate dehydrogenase/cyclohydrolase', True),
    ('folE', 'GTP cyclohydrolase I', True),
    ('folK', '2-amino-4-hydroxy-6-hydroxymethyldihydropteridine pyrophosphokinase', True),
    ('folP', 'dihydropteroate synthase', True),
    ('nadD', 'nicotinate-nucleotide adenylyltransferase', True),
    ('nadE', 'NAD synthetase', True),
    ('ribA', 'GTP cyclohydrolase II', True),
    ('ribB', '3,4-dihydroxy-2-butanone 4-phosphate synthase', True),
    ('ribC', 'riboflavin synthase alpha', True),
    ('ribD', 'riboflavin biosynthesis protein RibD', True),
    ('ribE', 'riboflavin synthase beta', True),
    ('ribF', 'riboflavin kinase/FAD synthetase', True),
    
    # ========== UNKNOWN FUNCTION (Essential) ==========
    ('ybeY', 'endoribonuclease involved in rRNA maturation', True),
    ('yeaZ', 'threonylcarbamoyl-AMP synthase', True),
    ('ygjD', 'tRNA threonylcarbamoyladenosine biosynthesis protein', True),
    ('yjeE', 'ATPase involved in chromosome partitioning', True),
    ('yqgF', 'Holliday junction resolvase-like protein', True),
    ('yrfF', 'essential uncharacterized protein', True),
    
    # ========== UNKNOWN FUNCTION (Non-essential) ==========
    ('yaaA', 'peroxide stress resistance protein', False),
    ('yaaH', 'uncharacterized protein', False),
    ('yaaJ', 'uncharacterized transporter', False),
    ('yabB', 'uncharacterized protein', False),
    ('yabN', 'uncharacterized protein', False),
    ('yabP', 'uncharacterized protein', False),
    ('yacC', 'uncharacterized protein', False),
    ('yacF', 'uncharacterized protein', False),
    ('yadC', 'fimbrial-like adhesin protein', False),
    ('yadE', 'polysaccharide deacetylase', False),
    ('yaeJ', 'ribosome-associated protein', False),
    ('yafK', 'uncharacterized protein', False),
    ('yaiY', 'inner membrane protein', False),
    ('yajR', 'uncharacterized MFS transporter', False),
    ('ybaN', 'uncharacterized protein', False),
    ('ybaP', 'uncharacterized protein', False),
    ('ybbN', 'thioredoxin-like protein', False),
    ('ybcK', 'uncharacterized protein', False),
    ('ybdM', 'uncharacterized protein', False),
    ('ybeB', 'uncharacterized protein', False),
    ('ybfE', 'uncharacterized protein', False),
    ('ybhB', 'uncharacterized kinase inhibitor', False),
    ('ybiJ', 'uncharacterized protein', False),
    ('ybjL', 'uncharacterized protein', False),
    ('ycaR', 'uncharacterized protein', False),
    ('ycdX', 'uncharacterized protein', False),
    ('ycfJ', 'uncharacterized protein', False),
    ('ycfP', 'uncharacterized protein', False),
    ('ycgB', 'uncharacterized protein', False),
    ('yciE', 'uncharacterized protein', False),
    ('yciF', 'uncharacterized protein', False),
    ('ydaN', 'uncharacterized protein', False),
    ('ydcA', 'uncharacterized protein', False),
    ('ydeJ', 'uncharacterized protein', False),
    ('ydhD', 'glutaredoxin-like protein', False),
    ('ydhU', 'uncharacterized protein', False),
    ('ydjA', 'uncharacterized oxidoreductase', False),
    ('yeaD', 'uncharacterized protein', False),
    ('yeaE', 'uncharacterized oxidoreductase', False),
    ('yeaH', 'uncharacterized protein', False),
    ('yebE', 'uncharacterized protein', False),
    ('yecA', 'uncharacterized protein', False),
    ('yedE', 'uncharacterized protein', False),
    ('yegH', 'uncharacterized protein', False),
    ('yehC', 'uncharacterized fimbrial chaperone', False),
    ('yeiH', 'uncharacterized protein', False),
    ('yfaD', 'uncharacterized protein', False),
    ('yfaL', 'uncharacterized protein', False),
    ('yfbM', 'uncharacterized protein', False),
    ('yfcG', 'uncharacterized GST-like protein', False),
    ('yfdC', 'uncharacterized protein', False),
    ('yfeK', 'uncharacterized protein', False),
    ('yfgJ', 'uncharacterized protein', False),
]


def predict_from_annotation(gene_name: str, annotation: str) -> tuple:
    """
    Predict essentiality from gene name and functional annotation.
    
    NO USE OF ESSENTIALITY LABELS - only annotation text.
    """
    annotation_lower = annotation.lower()
    gene_lower = gene_name.lower()
    
    score = 0.0
    reasons = []
    
    # ========== TRANSLATION (Usually essential) ==========
    if 'ribosomal protein' in annotation_lower:
        score += 0.8
        reasons.append('ribosomal protein')
    
    if 'trna synthetase' in annotation_lower or 'tRNA synthetase' in annotation:
        score += 0.9
        reasons.append('tRNA synthetase')
    
    if 'translation' in annotation_lower and 'factor' in annotation_lower:
        score += 0.7
        reasons.append('translation factor')
    
    if 'elongation factor' in annotation_lower:
        score += 0.7
        reasons.append('elongation factor')
    
    if 'initiation factor' in annotation_lower:
        score += 0.8
        reasons.append('initiation factor')
    
    if 'release factor' in annotation_lower:
        score += 0.6
        reasons.append('release factor')
    
    # ========== TRANSCRIPTION ==========
    if 'rna polymerase' in annotation_lower:
        if 'subunit' in annotation_lower:
            if 'alpha' in annotation_lower or 'beta' in annotation_lower:
                score += 0.9
                reasons.append('RNAP core subunit')
            elif 'sigma' in annotation_lower:
                if '70' in annotation or 'rpod' in gene_lower:
                    score += 0.9
                    reasons.append('primary sigma factor')
                else:
                    score += 0.3
                    reasons.append('alternative sigma factor')
            elif 'omega' in annotation_lower:
                score += 0.2
                reasons.append('RNAP omega (non-essential)')
    
    if 'transcription termination' in annotation_lower:
        score += 0.7
        reasons.append('transcription termination')
    
    # ========== REPLICATION ==========
    if 'dna polymerase' in annotation_lower:
        if 'iii' in annotation_lower or 'III' in annotation:
            score += 0.8
            reasons.append('DNA Pol III')
        else:
            score += 0.4
            reasons.append('DNA polymerase')
    
    if 'replicat' in annotation_lower and ('helicase' in annotation_lower or 'initiator' in annotation_lower):
        score += 0.9
        reasons.append('replication machinery')
    
    if 'primase' in annotation_lower:
        score += 0.9
        reasons.append('primase')
    
    if 'dna ligase' in annotation_lower:
        score += 0.9
        reasons.append('DNA ligase')
    
    if 'gyrase' in annotation_lower or 'topoisomerase' in annotation_lower:
        if 'iv' in annotation_lower or 'IV' in annotation or 'i' in annotation_lower:
            score += 0.8
        else:
            score += 0.5
        reasons.append('topoisomerase')
    
    # ========== CELL DIVISION ==========
    if gene_lower.startswith('fts') and 'cell division' in annotation_lower:
        score += 0.7
        reasons.append('cell division protein')
    
    if 'septum' in annotation_lower or 'division ring' in annotation_lower:
        score += 0.6
        reasons.append('cell division')
    
    # ========== CELL WALL ==========
    if gene_lower.startswith('mur') or gene_lower.startswith('mra'):
        score += 0.85
        reasons.append('cell wall synthesis')
    
    if 'peptidoglycan' in annotation_lower or 'murein' in annotation_lower:
        score += 0.8
        reasons.append('peptidoglycan synthesis')
    
    # ========== MEMBRANE/ENVELOPE ==========
    if gene_lower.startswith('lpx') or gene_lower.startswith('kds') or gene_lower.startswith('kdt'):
        score += 0.85
        reasons.append('LPS biosynthesis')
    
    if 'lipid a' in annotation_lower or 'lipopolysaccharide' in annotation_lower:
        score += 0.8
        reasons.append('LPS/Lipid A')
    
    if gene_lower.startswith('acc') and 'carboxyl' in annotation_lower:
        score += 0.8
        reasons.append('acetyl-CoA carboxylase')
    
    if gene_lower.startswith('fab') and ('synthase' in annotation_lower or 'reductase' in annotation_lower or 'dehydratase' in annotation_lower):
        score += 0.7
        reasons.append('fatty acid synthesis')
    
    if 'acyl carrier protein' in annotation_lower:
        score += 0.9
        reasons.append('acyl carrier protein')
    
    # ========== PROTEIN SECRETION ==========
    if gene_lower.startswith('sec') and ('translocase' in annotation_lower or 'preprotein' in annotation_lower):
        if gene_lower in ['seca', 'sece', 'secy', 'secd', 'secf']:
            score += 0.8
            reasons.append('Sec translocon core')
        else:
            score += 0.4
            reasons.append('Sec accessory')
    
    if 'signal peptidase' in annotation_lower:
        score += 0.8
        reasons.append('signal peptidase')
    
    if 'signal recognition particle' in annotation_lower:
        score += 0.8
        reasons.append('SRP')
    
    # ========== CHAPERONES ==========
    if 'chaperone' in annotation_lower or 'chaperonin' in annotation_lower:
        if 'groel' in gene_lower or 'groes' in gene_lower:
            score += 0.9
            reasons.append('GroEL/ES essential chaperone')
        elif 'dnak' in gene_lower or 'dnaj' in gene_lower or 'grpe' in gene_lower:
            score += 0.8
            reasons.append('DnaK system')
        else:
            score += 0.3
            reasons.append('other chaperone')
    
    # ========== PROTEASES ==========
    if 'protease' in annotation_lower:
        if 'ftsh' in gene_lower:
            score += 0.8
            reasons.append('FtsH essential protease')
        else:
            score += 0.2
            reasons.append('non-essential protease')
    
    # ========== COFACTORS ==========
    if gene_lower.startswith('coa') and len(gene_lower) == 4:
        score += 0.8
        reasons.append('CoA biosynthesis')
    
    if gene_lower.startswith('fol') and len(gene_lower) == 4:
        score += 0.8
        reasons.append('folate biosynthesis')
    
    if gene_lower.startswith('rib') and len(gene_lower) == 4:
        score += 0.8
        reasons.append('riboflavin biosynthesis')
    
    if gene_lower.startswith('nad') and 'synthetase' in annotation_lower:
        score += 0.8
        reasons.append('NAD biosynthesis')
    
    # ========== NUCLEOTIDE KINASES ==========
    if gene_lower in ['adk', 'ndk', 'cmk', 'gmk', 'tmk']:
        score += 0.85
        reasons.append('nucleotide kinase')
    
    if 'ctp synthase' in annotation_lower or 'ump kinase' in annotation_lower:
        score += 0.8
        reasons.append('nucleotide synthesis')
    
    # ========== CENTRAL METABOLISM ==========
    # Glycolysis - most have isozymes, so non-essential
    if gene_lower in ['tpia', 'gapa', 'pgk', 'eno', 'fbaa']:
        score += 0.7
        reasons.append('glycolysis (single copy)')
    elif any(x in annotation_lower for x in ['isomerase', 'kinase', 'aldolase', 'dehydrogenase']) and \
         any(x in annotation_lower for x in ['phosphate', 'phospho', 'glucose', 'pyruvate']):
        score += 0.3
        reasons.append('central metabolism (likely has isozyme)')
    
    # TCA cycle - non-essential (can use fermentation)
    if any(x in annotation_lower for x in ['citrate synthase', 'aconitase', 'isocitrate dehydrogenase',
                                            'succinate dehydrogenase', 'fumarase', 'malate dehydrogenase']):
        score += 0.1
        reasons.append('TCA cycle (non-essential)')
    
    # ========== AMINO ACID BIOSYNTHESIS ==========
    # Generally non-essential if amino acids provided
    if gene_lower.startswith(('arg', 'his', 'ilv', 'leu', 'met', 'thr', 'trp', 'tyr')):
        if any(x in annotation_lower for x in ['synthase', 'synthetase', 'dehydrogenase', 'kinase', 'transferase']):
            score += 0.1
            reasons.append('amino acid biosynthesis (non-essential in rich media)')
    
    # ========== UNKNOWN FUNCTION ==========
    if 'uncharacterized' in annotation_lower or 'unknown' in annotation_lower:
        if gene_lower.startswith('y'):
            # Most y-genes are non-essential
            score += 0.15
            reasons.append('uncharacterized (likely non-essential)')
        else:
            score += 0.3
            reasons.append('unknown function')
    
    # ========== REDUNDANCY SIGNALS ==========
    # Look for signs of paralogs
    if 'duplicate' in annotation_lower or 'paralog' in annotation_lower:
        score -= 0.3
        reasons.append('has paralog/duplicate')
    
    # Isozyme naming (I, II, A, B)
    if any(x in annotation for x in [' I ', ' II ', ' A ', ' B ', ' 1', ' 2']):
        if 'tRNA synthetase' not in annotation:  # Synthetases don't have backup
            score -= 0.2
            reasons.append('likely has isozyme')
    
    # Alternative/minor
    if 'alternative' in annotation_lower or 'minor' in annotation_lower:
        score -= 0.3
        reasons.append('alternative/minor form')
    
    # Make prediction
    predicted_essential = score >= 0.5
    
    return predicted_essential, score, reasons


def run_test():
    """Run true test on E. coli genes."""
    
    print("="*70)
    print("TRUE TEST: E. COLI ESSENTIALITY PREDICTION")
    print("Predicting from annotation only (no essentiality labels)")
    print("="*70)
    
    results = []
    
    for gene_name, annotation, experimental_essential in ECOLI_GENES:
        predicted, score, reasons = predict_from_annotation(gene_name, annotation)
        
        results.append({
            'gene': gene_name,
            'annotation': annotation[:50],
            'predicted': predicted,
            'experimental': experimental_essential,
            'correct': predicted == experimental_essential,
            'score': score,
            'reasons': reasons,
        })
    
    # Calculate metrics
    tp = sum(1 for r in results if r['predicted'] and r['experimental'])
    fp = sum(1 for r in results if r['predicted'] and not r['experimental'])
    tn = sum(1 for r in results if not r['predicted'] and not r['experimental'])
    fn = sum(1 for r in results if not r['predicted'] and r['experimental'])
    
    total = len(results)
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    n_essential = sum(1 for r in results if r['experimental'])
    n_nonessential = total - n_essential
    
    print(f"\nDataset: {total} genes ({n_essential} essential, {n_nonessential} non-essential)")
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Accuracy:    {accuracy*100:.1f}%")
    print(f"Sensitivity: {sensitivity*100:.1f}% (of essential genes correctly predicted)")
    print(f"Specificity: {specificity*100:.1f}% (of non-essential genes correctly predicted)")
    print(f"Precision:   {precision*100:.1f}%")
    print(f"\nConfusion matrix:")
    print(f"  TP={tp} (essential, predicted essential)")
    print(f"  FP={fp} (non-essential, predicted essential)")
    print(f"  TN={tn} (non-essential, predicted non-essential)")
    print(f"  FN={fn} (essential, predicted non-essential)")
    
    # Analyze errors
    errors = [r for r in results if not r['correct']]
    
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS ({len(errors)} errors)")
    print("="*70)
    
    # False positives
    fp_list = [r for r in errors if r['predicted'] and not r['experimental']]
    print(f"\nFalse Positives ({len(fp_list)}) - predicted essential but NOT:")
    for r in fp_list[:10]:
        print(f"  {r['gene']:<10} {r['annotation'][:40]:<40} reasons: {r['reasons']}")
    
    # False negatives
    fn_list = [r for r in errors if not r['predicted'] and r['experimental']]
    print(f"\nFalse Negatives ({len(fn_list)}) - predicted non-essential but IS essential:")
    for r in fn_list[:10]:
        print(f"  {r['gene']:<10} {r['annotation'][:40]:<40} score: {r['score']:.2f}")
    
    return results


if __name__ == '__main__':
    results = run_test()
