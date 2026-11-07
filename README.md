# PolicyUQ-VLM — Summit Route to Uncertainty-Aware Prostate MRI

```
      benign  suspicious  REFER
         \        |        /
          \       |       /        <- three-way triage (Eq. 7)
        [ uncertainty decomposition ]   aleatoric + epistemic
                  |
        [ commit ]  o---o---o---o     <- policy traverses an attention path
                  |   (PPO agent over an evidence library of 1297 actions)
        [ fuse ] T2W  DWI  ADC        <- shared-weight 3D Swin + descriptor alignment
```

PolicyUQ-VLM reads a multiparametric prostate MRI study (T2-weighted, diffusion-weighted, and the
apparent diffusion coefficient map) and decides *how much to look* before committing to a clinically
significant prostate cancer (csPCa) call. A shared-weight 3D vision-language encoder projects the three
sequences into a radiology-descriptor space; a proximal-policy-optimized agent walks an attention path
through that evidence and commits when it has seen enough; a hierarchical uncertainty module then splits
the prediction into aleatoric and epistemic parts and routes equivocal cases to review. This repository
is the route map: every pitch below corresponds to a module you can run.

---

## The approach — getting on the rock

Pick one rope.

```
# pip
python -m venv .venv && source .venv/bin/activate
pip install -e .

# conda
conda env create -f environment.yml
conda activate policyuq-vlm
pip install -e .

# docker
docker build -t policyuq-vlm .
docker run --rm policyuq-vlm --help
```

Developer extras (`ruff`, `black`, `isort`, `mypy`, `pytest`) and the optional text encoder dependencies:

```
pip install -e ".[dev,text]"
```

---

## Pitch-by-pitch — the modules you clip into

| Pitch | Package | What it protects |
|-------|---------|------------------|
| 1 | `policyuq_vlm/terrain` | T2W/DWI/ADC volumes, resample-center-normalize, augmentation, the 87-descriptor PI-RADS vocabulary |
| 2 | `policyuq_vlm/recon` | shared-weight 3D Swin backbone, InfoNCE descriptor alignment (Eq. 1), cross-sequence fusion |
| 3 | `policyuq_vlm/route` | MDP state, GRU context update (Eq. 2), PPO agent over 1297 actions, GAE |
| 4 | `policyuq_vlm/hazard` | heteroscedastic aleatoric head (Eq. 5), epistemic stochastic passes (Eq. 6), three-way triage (Eq. 7) |
| 5 | `policyuq_vlm/ledger` | InfoNCE, terminal reward (Eq. 3), PPO clipped surrogate (Eq. 4) |
| 6 | `policyuq_vlm/cairns` | AUROC, AP, Sens@90, ECE, Brier, DeLong, bootstrap CI |
| 7 | `policyuq_vlm/ascent` | three-phase trainer (Algorithm 1), AdamW + cosine, atomic checkpoints, EMA, calibration target |
| 8 | `policyuq_vlm/descent` | benchmark, referral analysis, per-PI-RADS subgroups, reliability bins |

---

## Rack & protection — configuration

Configs live in `routebook/` as TOML and are loaded into frozen `attrs` schemas
(`policyuq_vlm/basecamp/config.py`). `main.toml` carries the manuscript settings exactly; ablation and
supplementary files hold only the keys they change and are merged onto a base:

```
python -m policyuq_vlm.summit train --config routebook/ablation_no_policy.toml --base routebook/main.toml
```

Key belay points in `main.toml`: AdamW `lr=3e-4` (cosine, 0.1x in phase 3), `weight_decay=1e-2`,
`batch_size=16`, grad clip 1.0, mixed precision over 4 ranks; PPO `clip=0.2`, `gamma=0.99`,
`gae_lambda=0.95`, entropy `0.01`, 4 epochs per collection round; reward `alpha=1.0`, `beta=0.5`,
`step_cost=0.02`; `M=20` aleatoric Monte-Carlo samples, `K=10` epistemic passes; volume `192x192x48`,
`N=432` tokens, `d=768`.

---

## The ascent — commands

```
# train (writes runs/policyuq.pt)
python -m policyuq_vlm.summit train --config routebook/main.toml

# evaluate a benchmark (Tables I-III + referral + per-PI-RADS)
python -m policyuq_vlm.summit evaluate --config routebook/main.toml --checkpoint runs/policyuq.pt

# triage a cohort (probability + benign/suspicious/refer)
python -m policyuq_vlm.summit infer --config routebook/main.toml --checkpoint runs/policyuq.pt

# export the fused-classification graph to ONNX
python -m policyuq_vlm.summit trace --config routebook/main.toml --out runs/policyuq.onnx
```

Multi-GPU and SLURM launches are in `approaches/launch_train.sh`. The dataset readers are not implemented, and the
model currently uses a deterministic descriptor embedding table rather than pretrained text embeddings.
The supplied code and configurations do not by themselves reproduce the reported results.

---

## Summit register — what the route tops out at

The following numbers are reported in the manuscript and have not been verified with this repository.
Dataset-specific readers and trained weights are required before evaluation.

| Benchmark | Command | AUROC | AP | Sens@90 | ECE | Brier |
|-----------|---------|-------|----|---------|-----|-------|
| PI-CAI | `evaluate --config routebook/main.toml` | 0.907 ± 0.006 | 0.779 ± 0.008 | 0.703 ± 0.013 | 0.038 ± 0.004 | 0.152 ± 0.005 |
| PROSTATEx | `evaluate --config routebook/supplementary_prostatex.toml --base routebook/main.toml` | 0.894 ± 0.011 | 0.751 ± 0.014 | 0.668 ± 0.019 | 0.043 ± 0.006 | 0.169 ± 0.007 |
| Prostate158 | `evaluate --config routebook/supplementary_prostate158.toml --base routebook/main.toml` | 0.886 | 0.733 | — | 0.049 | — |

Referral on PI-CAI (Table V): 11.3% referral rate, csPCa sensitivity 0.978, and a 22.3% reduction in
false-positive interventions versus a confidence-free baseline. Per-PI-RADS AUROC (Table VI) shows the
largest gain on equivocal PI-RADS 3 lesions.

Ablations (Table IV) drop in AUROC relative to the full route:

| Config | Δ AUROC |
|--------|---------|
| `ablation_no_vl_alignment.toml` | -2.4 |
| `ablation_no_policy.toml` | -1.8 |
| `ablation_no_uncertainty.toml` | -0.9 |
| `ablation_no_fusion.toml` | -3.3 |
| `ablation_no_vl_policy.toml` | -4.6 |
| `ablation_reinforce.toml` | -1.3 |
| `ablation_dpo.toml` | -1.6 |

> Dataset-specific readers are not implemented. Training and evaluation require integration of
> preprocessed data and appropriate trained weights.

---

## Load & oxygen — compute budget

| Phase | Hardware | Wall-clock (PI-CAI) |
|-------|----------|---------------------|
| 1 — VL pre-training (50 ep) | 4 x A100 40GB, mixed precision | ~8 h |
| 2 — supervised fine-tuning (30 ep) | 4 x A100 40GB | ~4 h |
| 3 — policy optimization (≤40 ep, early stop) | 4 x A100 40GB | ~14 h |
| inference | 1 x A100 | ~2.4 s / case (10 epistemic passes) |

These are not softened; the policy phase carries most of the cost because each case rolls out a variable
trajectory (mean length 6.3 steps) and 10 stochastic passes for the epistemic estimate.

---

## Caches on the mountain — datasets

| Dataset | Version / split | License | Access |
|---------|-----------------|---------|--------|
| PI-CAI | public training, n=1500 | CC BY-NC 4.0 | https://pi-cai.grand-challenge.org/DATA/ |
| PROSTATEx | n=204 / 330 lesions | CC BY 3.0 | https://www.cancerimagingarchive.net/collection/prostatex/ |
| Prostate158 | n=158 (139/19) | not stated on challenge page | https://prostate158.grand-challenge.org/ |

Preprocess each into `data/<dataset>/` (resampled to 192x192x48, lesion bounding box, csPCa label) with
a dataset-specific reader. `approaches/prepare_data.sh` creates the expected directories; it does not
download or preprocess data. PI-CAI imaging is mirrored on Zenodo (DOI 10.5281/zenodo.6624726);
PROSTATEx is on TCIA (DOI 10.7937/K9TCIA.2017.MURS5CL).

---

## Crevasse checks — tests

```
make test      # full pytest suite
make lint      # ruff + black + isort
make type      # mypy --strict
```

The suite covers model components, metric calculations, reward behavior, and triage rules.

---

## Leave no trace — intended use and ethics

This is research software for retrospective, de-identified prostate MRI cohorts. It is not a medical
device and must not drive clinical decisions without prospective validation and regulatory clearance.
The referral mechanism is calibrated to keep csPCa sensitivity at or above 0.95 on a held-out set;
operating points must be re-fit per site. All datasets above are public and de-identified; respect each
license, in particular the non-commercial terms on PI-CAI.
