# 2026-10-23 · Cloud GPU compute: L4 spot in US regions

## What

Set up Google Cloud as the training compute for TernaVLA, alongside the Kaggle T4s named in the
plan: command-line access, project, billing and GPU quota checked; a GPU choice and a spending
order fixed. No training has run yet and nothing was spent in this step.

## Why

The training ladder in [PLAN.md](../PLAN.md) was sized for Kaggle's 2 × T4, which comes with a
weekly GPU quota. The ternary-LoRA learning-rate sweep on M, full LIBERO evaluation and
simulator data generation would take weeks of that quota. Promotional cloud credits make those
steps practical without paying for compute.

GPU options considered:

| Option | Approx. cost | Verdict |
|---|---|---|
| L4 24 GB, spot, `g2-standard-4`, US region | ₹25–35 per hour | **chosen** |
| L4, Mumbai region | similar | rejected: spot capacity often unavailable |
| H100 80 GB, on demand | ~₹950 per hour | rejected: credits would last about 5 hours |
| Kaggle 2 × T4 | free, weekly quota | kept for anything that fits it |

The L4 has 24 GB of memory, enough for every stage of the S tier and for the M tier with
LoRA, and costs about 30× less per hour than an H100. Spot machines can be stopped by Google at
any time, so every job must save checkpoints often and be able to resume.

## Result

- Budget: about ₹3,000 of credits on the first account, then about ₹14,000 on a collaborator's
  account once the first runs out. At ₹25–35 per hour that is roughly 85–120 L4-hours first and
  about 500 L4-hours in total, minus disk storage.
- Quota: one GPU at a time across all regions, and one spot L4 in each US region checked
  (us-central1, us-east1, us-east4, us-east5, us-west1, us-west4, us-south1). If us-central1
  runs out of spot capacity, the job moves to another US region.
- An existing stopped spot L4 machine (us-central1-a, PyTorch 2.9 / CUDA 12.9 deep-learning
  image, 100 GB standard disk) is reused rather than creating a new one. Its disk is billed even
  while the machine is stopped.
- Rules for spending: L4, spot, US regions only; checkpoint and resume on every job; stop the
  machine when a job ends; keep disks small and delete them when no longer needed.

## Effect on the paper

The plan's first contribution is "adaptation on a free-tier budget (Kaggle T4s)". Using cloud
credits changes that claim. Every run will report its GPU type and GPU-hours so the cost of
reproducing it is explicit; whether the headline recipe is still restricted to free-tier hardware
is an open decision.

## Next

- Write the S-tier training pipeline: cached SigLIP2 features, the 13-query action head, and the
  LIBERO-Spatial data loader. Nothing GPU-side can start before this exists.
- Set budget alerts on the billing account before the first run.
- First GPU run: S-tier head-only probe on LIBERO-Spatial, gate ≥ 90 % (PLAN.md, weeks 1–2).
