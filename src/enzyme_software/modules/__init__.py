ModuleMinus1SRE = None
Module0StrategyRouter = None
Module1TopoGate = None
Module2ActiveSiteRefinement = None
Module3ExperimentDesigner = None

try:
    from enzyme_software.modules.module_minus1_reactivity_hub import ModuleMinus1SRE
except Exception:
    ModuleMinus1SRE = None

try:
    from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter
except Exception:
    Module0StrategyRouter = None

try:
    from enzyme_software.modules.module1_topogate import Module1TopoGate
except Exception:
    Module1TopoGate = None

try:
    from enzyme_software.modules.module2_active_site_refinement import (
        Module2ActiveSiteRefinement,
    )
except Exception:
    Module2ActiveSiteRefinement = None

try:
    from enzyme_software.modules.module3_experiment_designer import (
        Module3ExperimentDesigner,
    )
except Exception:
    Module3ExperimentDesigner = None

__all__ = [
    name
    for name, value in {
        "ModuleMinus1SRE": ModuleMinus1SRE,
        "Module0StrategyRouter": Module0StrategyRouter,
        "Module1TopoGate": Module1TopoGate,
        "Module2ActiveSiteRefinement": Module2ActiveSiteRefinement,
        "Module3ExperimentDesigner": Module3ExperimentDesigner,
    }.items()
    if value is not None
]
