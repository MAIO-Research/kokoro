# Project Memory

This is a fork of kokoro / StyleTTS2 being retargeted from the stock
LJSpeech (English) recipe to train a **single-speaker Armenian ("hy") voice**
on the `telcell_female` dataset.

## What changed vs. upstream

**Data pipeline**
- [Data/prepare_hy_lists.py](Data/prepare_hy_lists.py) — one-off script converting
  `telcell_female`'s `audio|text|speaker|language` files (at
  `/mnt/filesystem-m3/data/telcell_female`) into kokoro's `audio|text|speaker`
  list format. Produces [Data/train_list_hy.txt](Data/train_list_hy.txt) (5420 lines)
  and [Data/val_list_hy.txt](Data/val_list_hy.txt) (54 lines). Text is IPA-phonemized
  Armenian, single speaker id `0`. Original `Data/train_list.txt`/`val_list.txt`
  (LJSpeech) are left untouched.

**Config ([Configs/config.yml](Configs/config.yml))**
- `log_dir` → `Models/telcell_female_hy`
- `ASR_config`/`ASR_path` point at a separately-trained Armenian AuxiliaryASR
  checkpoint (absolute path under `/mnt/filesystem-m3/workspace/armen/AuxiliaryASR/Checkpoint/`)
- `PLBERT_dir` removed — no longer used (see PL-BERT replacement below)
- `data_params.train_data`/`val_data` point at the new hy lists;
  `root_path` → `/mnt/filesystem-m3/data/telcell_female/wavs`
- `OOD_data` set to `train_list_hy.txt` as a placeholder (no separate OOD
  corpus exists for Armenian; `use_ind` is now forced `True` in
  `train_second.py` so OOD text is never actually sampled — this is just to
  satisfy the dataloader's plumbing)
- `n_token: 178 → 179` (new symbol table has one more token — added `<sos>`,
  `<eos>`, `<unk>`, changed `_pad` from `$` to `_`)
- `bert_lr: 0.00001 → 0.0001` (now training the PL-BERT replacement from
  scratch jointly with the rest of the model, not fine-tuning a pretrained
  checkpoint)

**Text symbols ([text_utils.py](text_utils.py))**
- Symbol table matches the Armenian AuxiliaryASR aligner's vocab so phoneme
  indices line up with the pretrained ASR/aligner embeddings
- `meldataset.py`'s duplicated `TextCleaner`/symbol table was removed in
  favor of importing from `text_utils.py` (dedup)

**PL-BERT replacement ([Utils/PLBERT/util.py](Utils/PLBERT/util.py))**
- No pretrained PL-BERT exists for Armenian, so the original ALBERT-based
  loader (loads a pretrained checkpoint from `PLBERT_dir`) was replaced with
  `PlainTextBERT`: a from-scratch CNN+LSTM `TextEncoder` (reuses the model's
  own text encoder architecture), trained jointly with the rest of the model
  instead of loaded from a checkpoint. Signature changed from
  `load_plbert(log_dir)` to `load_plbert(model_params)`.

**Training scripts ([train_first.py](train_first.py), [train_second.py](train_second.py))**
- Both call `load_plbert(model_params)` instead of `load_plbert(BERT_path)`
  (reordered so `model_params` is built before the PL-BERT load)
- `train_second.py`: removed the random 50/50 `use_ind` choice between
  in-distribution and OOD reference text — now always `use_ind = True`,
  since there's no OOD corpus for this dataset

**requirements.txt**: added `phonemizer`, `ipykernel`, `pandas`

## Config semantics (confirmed from code)

- `save_freq` (Configs/config.yml) → used as `saving_epoch`; checkpoints
  saved every N **epochs**: `if epoch % saving_epoch == 0`
  (train_second.py:767, similarly train_first.py)
- `log_interval` → every N **iterations/steps** within an epoch:
  `if (i+1) % log_interval == 0` — controls loss printing/logging cadence

## How to run training

First stage (uses `accelerate`):
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
If accelerate isn't configured yet in this environment: run `accelerate config`
first, or do a quick single-GPU fp16 run with
`accelerate launch --mixed_precision=fp16 --num_processes=1 train_first.py --config_path ./Configs/config.yml`.

Second stage (plain python, no accelerate). This environment uses **`uv`** —
run with `uv run python`, not bare `python` (there is no system torch):
```bash
uv run python train_second.py --config_path Configs/config_2nd.yml
```

- Second stage uses its own config [Configs/config_2nd.yml](Configs/config_2nd.yml)
  (separate from the first-stage `config.yml`), with:
  - `first_stage_path` = `Models/telcell_female_hy/epoch_1st_00160.pth`
  - `log_dir` = `Models/telcell_female_hy_2nd` (checkpoints `epoch_2nd_%05d.pth`
    every 10 epochs)
  - `diff_epoch: 0` and `joint_epoch: 0` — jumps straight into joint training +
    the mpd/msd GAN from epoch 0 (so Disc/Gen losses are non-zero from step 1;
    this is intentional, not a bug). Note `train_second.py` here has had the
    diffusion sampler and SLM-adversarial training stripped out — there is no
    style diffusion; preview/eval style is taken from the utterance's own GT mel.
- Do NOT `pkill -f "train_second.py"` from a shell whose own command line
  contains that string — `pkill -f` matches the killer process itself. Use the
  bracket trick: `pkill -9 -f "[t]rain_second"`.

(Upstream README also documents `train_finetune.py` /
`train_finetune_accelerate.py` for fine-tuning, not yet adapted here.)

## Fixed bug: silent DataParallel weight-load failure → stage-2 NaN

[models.py](models.py) `load_checkpoint` originally did
`model[key].load_state_dict(params[key], strict=False)`. In
[train_second.py](train_second.py) every module is wrapped in `MyDataParallel`
**before** `load_checkpoint` runs, so the module's keys are `module.`-prefixed
while the checkpoint's are not. With `strict=False` this **silently loaded
nothing** — `style_encoder`/`decoder`/`text_encoder` kept their random init
(you still saw `"<module> loaded"`). Random-init `StyleEncoder` (spectral_norm)
explodes to ~1e19; that style vector overflows the AdaIN blocks in
`predictor.F0Ntrain` → `F0_fake`/`N_fake` = NaN → NaN decoder audio → NaN mel
loss → the old `set_trace()` NaN trap dropped the run into `ipdb`.
(`pitch_extractor`/`text_aligner` were fine only because they're loaded
separately, before the DataParallel wrap.)

**Fix (applied):** `load_checkpoint` now unwraps `DataParallel`
(`model[key].module`) before `load_state_dict`, and prints a WARNING if any
keys are missing/unexpected instead of failing silently. Also replaced the
interactive `set_trace()` NaN trap in `train_second.py` with a diagnostic
print + skip-batch so an unattended run won't hang in `ipdb`. Verified: modules
load with 0 warnings; training runs NaN-free with losses decreasing normally.
