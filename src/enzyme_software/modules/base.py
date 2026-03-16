from __future__ import annotations

from enzyme_software.context import PipelineContext


class BaseModule:
    name = "Base Module"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError
