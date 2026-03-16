from __future__ import annotations

from enzyme_software.modules.module2_active_site_refinement import _apply_variant_energy_scoring


def test_variant_delta_delta_improves_k():
    variants = [
        {
            "variant_id": "V_improve",
            "delta_deltaG_dagger_kJ_per_mol": -3.0,
            "strain_displacement_A": 0.0,
            "score": 0.0,
        },
        {
            "variant_id": "V_base",
            "delta_deltaG_dagger_kJ_per_mol": 0.0,
            "strain_displacement_A": 0.0,
            "score": 0.0,
        },
    ]
    selected = {"delta_g_mean": 20.0}
    job_card = {
        "reaction_condition_field": {"given_conditions": {"temperature_c": 30.0}},
        "condition_profile": {},
    }
    scored = _apply_variant_energy_scoring(variants, selected, job_card)
    variant_map = {item["variant_id"]: item for item in scored}
    assert variant_map["V_improve"]["k_variant_s_inv"] > variant_map["V_base"]["k_variant_s_inv"]


def test_strain_penalty_can_reverse():
    variants = [
        {
            "variant_id": "V_strain",
            "delta_deltaG_dagger_kJ_per_mol": -3.0,
            "strain_displacement_A": 3.0,
            "score": 0.0,
        },
        {
            "variant_id": "V_base",
            "delta_deltaG_dagger_kJ_per_mol": 0.0,
            "strain_displacement_A": 0.0,
            "score": 0.0,
        },
    ]
    selected = {"delta_g_mean": 20.0}
    job_card = {
        "reaction_condition_field": {"given_conditions": {"temperature_c": 30.0}},
        "condition_profile": {},
    }
    scored = _apply_variant_energy_scoring(variants, selected, job_card)
    variant_map = {item["variant_id"]: item for item in scored}
    assert variant_map["V_strain"]["k_variant_s_inv"] < variant_map["V_base"]["k_variant_s_inv"]
