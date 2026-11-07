# Implementation Map

Paper provenance lives here only; source files carry no comments and no docstrings.

| paper section | eq / fig / table | file | module | notes |
|---|---|---|---|---|
| §III-B.1 3D Vision Backbone | — | `policyuq_vlm/recon/swin3d.py` | `Swin3DEncoder` | shared-weight 3D Swin; patch 4x4x2, window 6x6x6, stages [96,192,384,768]; 192x192x48 -> N=432 tokens, d=768 |
| §III-B.2 Radiology Descriptor Alignment | Eq.(1) | `policyuq_vlm/recon/alignment.py`, `policyuq_vlm/ledger/vision_language.py` | `DescriptorAligner`, `vision_language_loss` | InfoNCE over 87 frozen descriptor embeddings, tau=0.07; z_i = Avg(F_i) |
| §III-B.2 descriptor vocabulary | — | `policyuq_vlm/terrain/vocabulary.py` | `DESCRIPTORS`, `build_vocabulary` | 87 PI-RADS descriptors: 12 sectors, T2/DWI grades, ADC, shape/margin/size, EPE/SVI |
| §III-B.3 Cross-Sequence Fusion | — | `policyuq_vlm/recon/fusion.py` | `CrossSequenceFusion` | concat 3 token sets + [CLS]; 4-layer 8-head cross-attention transformer -> h |
| §III-B encoder (text) | — | `policyuq_vlm/recon/text_encoder.py` | `FrozenDescriptorEncoder` | PubMedBERT (frozen) when available, else deterministic frozen descriptor embedding |
| §III-C.1 MDP Formulation | Eq.(2) | `policyuq_vlm/route/context.py` | `ContextGRU` | s_t=[h;c_t;u_t]; c_{t+1}=GRU(c_t, f_{a_t}); 2-layer 512 |
| §III-C.1 action space | — | `policyuq_vlm/route/state.py` | `EvidenceLibrary`, `RouteState` | |A| = 1297 = 3 x 432 + 1 commit; token feature lookup f_{a_i} |
| §III-C.1 policy / value heads | — | `policyuq_vlm/route/agent.py` | `PolicyAgent` | 3-layer MLP policy head over actions; 2-layer MLP value V_psi(s_t) |
| §III-C.1 rollout | — | `policyuq_vlm/route/trajectory.py` | `rollout`, `Trajectory` | discrete actions until Commit; mean length ~6.3 |
| §III-C.2 Reward Design | Eq.(3) | `policyuq_vlm/ledger/reward.py` | `terminal_reward` | R_T = alpha*1[yhat=y*] - beta*|p_pred - p_ideal| - lambda*T ; alpha=1.0,beta=0.5,lambda=0.02 |
| §III-C.2 calibration target | — | `policyuq_vlm/ascent/calibration.py` | `CalibrationTarget` | p_ideal from 15% held-out calibration subset; refreshed every 5 PPO epochs |
| §III-C.3 PPO Training | Eq.(4) | `policyuq_vlm/ledger/ppo.py`, `policyuq_vlm/route/advantage.py` | `ppo_surrogate`, `generalized_advantage` | clip eps=0.2; GAE gamma=0.99 lambda=0.95; entropy eta=0.01; 4 epochs/round |
| §III-D.1 Aleatoric (heteroscedastic) | Eq.(5) | `policyuq_vlm/hazard/aleatoric.py` | `HeteroscedasticHead`, `heteroscedastic_loss` | log-variance head; logit + sigma*eps; M=20 MC; cross-entropy context |
| §III-D.2 Epistemic (stochastic passes) | Eq.(6) | `policyuq_vlm/hazard/epistemic.py` | `epistemic_variance` | K=10 policy-conditioned stochastic passes; sigma^2_epi = var, p_bar = mean |
| §III-D.2 fusion of uncertainty | — | `policyuq_vlm/hazard/decomposition.py` | `decompose` | sigma^2_total = sigma^2_ale + sigma^2_epi |
| §III-D.3 Selective Referral | Eq.(7) | `policyuq_vlm/hazard/triage.py` | `triage_decision`, `select_threshold` | benign / suspicious / refer; delta s.t. csPCa sensitivity >= 0.95 |
| §III-E Algorithm 1 | Alg.1 | `policyuq_vlm/ascent/trainer.py` | `ThreePhaseTrainer` | Phase1 VL pretrain 50ep; Phase2 supervised fine-tune 30ep; Phase3 PPO <=40ep early-stop |
| §III-E phase-3 objective | — | `policyuq_vlm/ascent/trainer.py` | `_phase3_step` | L = L_PPO + L_V + mu*L_het, mu=0.3; LR at 0.1x |
| §IV-C optimizer / schedule | — | `policyuq_vlm/ascent/optim.py` | `build_optimizer`, `build_scheduler` | AdamW lr=3e-4 cosine, wd=1e-2, grad clip 1.0 |
| §IV-C DDP / AMP / ckpt / EMA / seed | — | `policyuq_vlm/ascent/runtime.py`, `policyuq_vlm/basecamp/seeding.py` | `distributed_context`, `atomic_save`, `Ema`, `set_seed` | atomic checkpoint write; seed stored + restored |
| Table I / II / III | — | `policyuq_vlm/descent/benchmark.py` | `evaluate_benchmark` | AUROC, AP, Sens@90, ECE, Brier per dataset |
| Table IV | — | `routebook/ablation_*.toml` | — | one config per ablation |
| Table V | — | `policyuq_vlm/descent/referral.py` | `referral_analysis` | ref rate, sensitivity, false-positive reduction; aleatoric-only / epistemic-only |
| Table VI | — | `policyuq_vlm/descent/subgroup.py` | `per_pirads_auc` | AUROC within each PI-RADS subgroup, bootstrap CI |
| §IV-B metrics | — | `policyuq_vlm/cairns/*.py` | `auroc`, `average_precision`, `sensitivity_at_specificity`, `expected_calibration_error`, `brier_score`, `delong_test`, `bootstrap_ci` | ECE 15 bins; DeLong (ref [41]); Sens@90 |
| §IV-A datasets / preprocessing | — | `policyuq_vlm/terrain/datasets.py`, `policyuq_vlm/terrain/preprocess.py` | `MpMriCase`, `resample_center_normalize` | 192x192x48 isotropic; flip, rot +/-15, elastic (sep 64, mag 10), Gaussian denoise sigma=0.05 |

## Code-availability statement (verbatim from manuscript)

The manuscript text contains no explicit code-availability statement (none present to quote).
