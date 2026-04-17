from __future__ import annotations


def station_from_trace_id(trace_id: str) -> str:
    """
    Extract station identifier (NET.STA) from a trace id.

    Examples
    --------
    - 'NET.STA.LOC.CHAN' -> 'NET.STA'
    - 'NET.STA' -> 'NET.STA'
    - 'STA' -> 'STA'
    """
    parts = str(trace_id).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(trace_id)

