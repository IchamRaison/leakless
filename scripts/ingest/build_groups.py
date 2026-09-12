"""Reconstruct group keys so the split cannot leak.

Grouping key, per docs/EVAL_PROTOCOL.md §2:
    material | region | pressure | flow rate | device | base id with _1/_2 suffix stripped

Known hazards (documented, not hypothetical):
  - numerous _1/_2 suffixes  -> several clips share one underlying event
  - hydrophone AND logger under identical conditions -> same event, two devices
  - 1-second clips from continuous sessions -> strong correlation between neighbours

The source note warns that filenames do NOT guarantee a perfect session
reconstruction. Where grouping stays ambiguous, this script must REPORT the
ambiguity rather than resolve it silently.

NOT IMPLEMENTED.
"""

raise NotImplementedError("stub - pending archive audit")
