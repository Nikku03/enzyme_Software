from enzyme_software.modules.module2_active_site_refinement import _build_mechanism_spec


def test_mechanism_spec_exploratory_mismatch():
    spec = _build_mechanism_spec(
        reaction_family="hydrolysis",
        route_label="serine_hydrolase",
        candidate_residues_by_role={
            "nucleophile": ["Cys217"],
            "base": ["His104"],
            "acid": ["Asp138"],
        },
        mechanism_policy="exploratory",
    )
    assert spec["policy_action"] == "REQUEST_DISAMBIGUATION"
    assert spec["mismatch_reason"] is not None


def test_mechanism_spec_strict_mismatch():
    spec = _build_mechanism_spec(
        reaction_family="hydrolysis",
        route_label="serine_hydrolase",
        candidate_residues_by_role={
            "nucleophile": ["Cys217"],
            "base": ["His104"],
            "acid": ["Asp138"],
        },
        mechanism_policy="strict",
    )
    assert spec["policy_action"] == "SWITCH_ROUTE"
    assert spec["mismatch_reason"] is not None
