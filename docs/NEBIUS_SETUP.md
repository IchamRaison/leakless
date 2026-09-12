# NEBIUS_SETUP

**Nebius compute setup follows the private hackathon organizer guide. Contact organisers for access.**

---

## Why this file is short

The organiser's setup guide is marked **"Internal — do not distribute"** and states that it
must only be used when the hackathon organiser has approved it. Its content is therefore
**deliberately not reproduced in this repository** — not the billing route, not the
organiser-supplied billing identity, not the promo-code procedure.

Team members who need the procedure should take it from the organiser's own posting in the
event Discord, or ask the Nebius engineers present on site.

---

## What can be said here

- Compute for this hackathon runs on **Nebius**, with a sponsored voucher per team.
- The console is at `console.nebius.com`.
- **Serverless AI** is the recommended path: `Jobs` for training and batch work (billing stops
  when the job ends), `Endpoints` for serving a demo (**billing continues until stopped** —
  keep them off except during a demo).
- **GPU sanity check**: the console's Job card has a built-in **`Run nvidia-smi`** action.
  Open the job, read the Logs; a GPU listed means compute is live. That is the only
  verification we run before the team validates a subject.
- If a launch fails for capacity reasons, the console exposes a capacity view per region and
  preset, and preemptible instances as a fallback. Details in the organiser's guide.
- **Credit hygiene**: set a timeout on every job, stop idle endpoints, watch usage in Billing,
  and right-size — one GPU is enough for most fine-tunes and demos.

---

## Secrets

**No promo code, voucher, token or API key belongs in this repository** — not in files, not in
notebooks, not in commit messages, not in shell history. They are entered by hand, in the
console, by the account owner.

---

## Public references

| | |
|---|---|
| Nebius console | `console.nebius.com` |
| Nebius for Startups | `nebius.com/startups` |
| Research grants | `nebius.com/nebius-research-grants` |
| Builder program | `dev.nebius.com/builders` |
| Fellows | `nebius.com/fellows` |

*(These are public Nebius programme pages, unrelated to the event's internal setup.)*
