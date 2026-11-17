"""Do the geometric part of the WFSS contamination modeling."""

__all__ = ["WfssGeomStep"]


def __getattr__(name):
    if name == "WfssGeomStep":
        from .wfss_geom_step import WfssGeomStep

        return WfssGeomStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
