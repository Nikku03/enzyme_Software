from computational_engines import XTBEngine, VinaEngine, OpenMMEngine, check_engines

# Verify installation
print(check_engines())

# Dock ibuprofen into CYP2C9
vina = VinaEngine()
result = vina.dock("1OG5_prepared.pdbqt", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", pdb_id="1OG5")
print(result["topogate_scores"])