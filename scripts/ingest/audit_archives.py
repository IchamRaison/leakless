"""Audit the three Zenodo archives before any split is fixed.

Dataset: https://zenodo.org/records/18631450  (CC BY 4.0)
  - leak acoustic data.rar          announced 500 clips  -- listed by Hicham
  - no leak acoustic data.rar       announced 386 clips  -- NEVER OPENED
  - environmental noise.rar         announced 114 clips  -- NEVER OPENED

Goal: replace "announced" with "verified" for all three, and report what the
filenames actually encode (material, region, pressure, flow rate, device).

NOT IMPLEMENTED. Nothing is downloaded until the team validates the subject.
See docs/DATASET_CARD.md and docs/HACKATHON_OPTIONS.md.
"""

raise NotImplementedError("stub - pending subject decision")
