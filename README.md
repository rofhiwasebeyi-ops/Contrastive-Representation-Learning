# Contrastive Representation Learning with Generative Replay for Smart-Home Routine Anomaly Detection

## The research question

Does generative replay — synthesising compact, learned stand-ins for past
days instead of storing them — preserve contrastive representation
stability and anomaly-detection reliability as well as real-buffer replay,
on real behavioural-sensor data, while requiring a fundamentally different
storage/privacy trade-off?

## What's compared, and why exactly this and nothing more

- **naive_finetuning** — lower-bound baseline: no memory of the past at all.
- **real_buffer_replay** — stores actual past days and replays them. The
  direct alternative to the proposed method; the whole point of testing
  generative replay is to see whether it can replace this.
- **generative_replay** (proposed) — a per-resident conditional VAE
  compresses routine history into a learned representation and synthesises
  replay days from it. No raw historical days are retained.
- **pca_personal_baseline** (sub-RQ4 baseline) — median-of-reference-window
  personal routine profile + PCA reconstruction error. The closest
  published prior approach conceptually (personal baseline + deviation
  scoring), used to show what the contrastive + generative-replay method
  adds over a simpler, non-continual, non-replaying statistical baseline.

## Data

Real CASAS smart-home data: **Aruba** (single resident, 220 days) and
**Milan** (single resident, 72 days), both from
https://zenodo.org/records/17180309 (`new_labeled_data.zip`) — see "Getting
real data in" below. Not bundled in this repo (large, and not ours to
redistribute).

## Results (current, reproducible via `run_all.py`)

**Aruba**: `generative_replay` beats `real_buffer_replay` in **5/5** hand-
picked hyperparameter configurations (mean AUROC 0.468 vs 0.425 across 3
seeds each).

**Milan**: **3/5** configurations — no consistent advantage (means nearly
identical, 0.536 vs 0.537). 

**Storage**: the VAE's fixed footprint (~85KB Aruba, ~79KB Milan) is
*larger* than even an unbounded real-data buffer across these datasets'
actual time spans (crossover point ~304 days Aruba / ~342 days Milan,
longer than either dataset runs). A day here is only a ~70-dim feature
vector, not an image, so the storage argument that motivates generative
replay in the image domain (e.g. ReplayCAD) doesn't transfer
proportionally to this low-dimensional tabular domain — worth stating
explicitly rather than assuming the image-domain argument carries over.

**Generated outputs** (`show_generated_samples.py`): the VAE generates
noticeably flatter, more homogeneous days than real ones — real days show
sharp activity spikes; generated samples stay in a tight band and look
similar to each other across different latent draws. This is a well-known
VAE behaviour (the KL term trades reconstruction sharpness for a smooth
latent space — mode-averaged/"blurry" outputs, the tabular analogue of
blurry image generation), not a bug, and a legitimate limitation to discuss
and propose future work against (e.g. a normalising flow or diffusion-based
generator instead).

## Project layout

```
src/
  data_loader.py            CASAS raw .txt parser -> tidy event DataFrame
  features.py                daily event aggregation -> fixed-length feature
                              vector (z-score standardised against a
                              reference window)
  encoder.py                  contrastive encoder (MLP) + NT-Xent loss +
                              augmentation
  generative_replay.py        conditional VAE (the proposed replay mechanism)
  anomaly.py                   routine prototype (deviation/anomaly scoring)
  strategies.py                the 3 compared strategies
  baselines.py                 PCA-personal-baseline (sub-RQ4)
  inject_deviations.py         synthetic deviation injection into real data
                              (the only source of ground-truth AUROC, since
                              real CASAS data has no true anomaly labels)
  train.py                     runs all 3 strategies + the baseline once,
                              prints the comparison table
  tune.py                      runs 5 hand-picked hyperparameter configs,
                              reports how many generative_replay wins
  show_generated_samples.py    generated-output examples (rubric requirement)
  plots.py                     generates report figures from saved JSON
data/
  raw/                        aruba.txt / milan.txt go here (not included)
```

## Running it (VS Code)

1. Open this folder as the workspace root.
2. Install `requirements.txt` (ideally in a venv).
3. **Run and Debug** (Ctrl+Shift+D) → pick a configuration. Each script has
   one, pre-filled with the exact arguments used for the reported results.
4. `run_all.py` reproduces everything in one go (~2-3 minutes).

## Running it (command line)

```
cd src
python train.py --home ../data/raw/aruba.txt --reference-days 14 \
    --inject-real-deviations 15 --injection-seed 1
python tune.py --home ../data/raw/aruba.txt --reference-days 14 \
    --n-deviations 15 --seeds 1,2,3
python show_generated_samples.py --home ../data/raw/aruba.txt
```

## Reproducing everything: `run_all.py`

```
python run_all.py     
```

Runs the full pipeline end-to-end (both homes, both with injected
deviations; the 5-config comparison on both homes; all figures including
the generated-output examples) and write everything to `outputs/`. Expected
runtime ~2-3 minutes.

## Live demo (`demo.py`)

For a quick presentation walkthrough — runs a fast, illustrative slice
live (~1.5 minutes), not the full pre-generated evidence:

```
python demo.py    # or the "Live Demo" launch config in VS Code
```

Part 1: a full real run on Aruba with injected deviations, live. Part 2: the
5-config comparison with 1 seed (not the full 3-seed result — that's in
`outputs/figures/config_comparison.png` already), just to prove the
comparison harness itself runs on request.

## Getting real data in

1. Download `new_labeled_data.zip` from https://zenodo.org/records/17180309
2. Unzip; copy `aruba.txt` and `milan.txt` into `data/raw/`
3. Run `python run_all.py`

## Figures (`outputs/figures/`, generated by `plots.py` / `show_generated_samples.py`)

- **`auroc_comparison.png`** — single-seed AUROC per strategy, both homes.
- **`storage_comparison.png`** — the capped/unbounded/VAE storage argument.
- **`config_comparison.png`** — the actual result: generative_replay vs.
  real_buffer_replay AUROC across the 5 hand-picked configs, per home. This
  is the figure that supports the "5/5 on Aruba, 3/5 on Milan" claim.
- **`generated_samples_routine_shape.png`** / **`generated_samples_heatmap.png`**
  — real vs. VAE-generated synthetic days (rubric: generated output examples).

All figures regenerate from `outputs/results/*.json` without re-running any
experiments — `python plots.py` alone is enough if the JSON files already
exist.