"""Measure what a naive split would have bought us -- and report the gap.

Per docs/EVAL_PROTOCOL.md §4, the gap IS a result:

    run A : naive random split, clip by clip   -> optimistic score
    run B : grouped by condition/session/device -> honest score
    report: A - B, with the split definition attached to each number

A score reported without its split definition is not a result.

NOT IMPLEMENTED.
"""

raise NotImplementedError("stub - pending grouping")
