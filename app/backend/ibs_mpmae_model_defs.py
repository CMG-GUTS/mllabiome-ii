"""Compatibility module for older IBS MPMA-E deployment artifacts.

Some exported ``ibs_mpma_e_model.zip`` files pickle deployment classes as
``ibs_mpmae_model_defs.*`` because the exporter loaded the sweep/model-defs file
under that module name. The inference-only app keeps the actual implementation
in ``app.backend.ibs_mpmae_runtime`` and registers this shim so unpickling works
without the original manuscript repository or the ``mllabiome`` package.
"""

try:
    from app.backend.ibs_mpmae_runtime import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct app/backend execution fallback
    from ibs_mpmae_runtime import *  # type: ignore  # noqa: F401,F403
