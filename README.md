# Digital Restoration of Indian Folk Paintings

A four-stage pipeline that turns a scan of a damaged folk painting (Madhubani, Pattachitra, Warli, Kalamkari) into a restored, high-resolution image. Alongside the image it saves a per-pixel confidence map that shows how much of each repaired area the model had to guess.

```
damaged painting
  └─ M1  CLAHE (LAB) + U-Net (pretrained ResNet34)  →  damage mask  [crack | loss | stain]
  └─ M2  LaMa (Fourier convolutions) inpainting, applied inside the mask only
  └─ M3  colour de-fading: Pix2Pix (residual U-Net + PatchGAN)  or  reference LAB transfer
  └─ M4  Real-ESRGAN x4 tiled super-resolution  +  spatial confidence map
```

## Why the previous version failed on real paintings, and what changed

| Problem | Fix in this version |
|---|---|
| Synthetic cracks (smooth Bézier strokes) looked nothing like real ones, so the model learned to detect the synthetic strokes rather than damage | `synth_damage.py`: polygonal **craquelure** networks, branching cracks of varying width, dark or light cracks with raised edges, flakes that show the paper/cloth ground (clustered along cracks), water stains with tide-lines, foxing, mold |
| Clean training tiles did not look like real scans | The damage is followed by scan degradation (uneven lighting, blur, noise, JPEG, resampling) |
| Painted linework looks like cracks | 25% of training tiles are undamaged, with empty masks, so the model learns that linework is not damage |
| Only synthetic data was used | **Hand-labelled real tiles** are mixed into training (30% of samples), and real damage is cut out and pasted onto clean tiles (`damage_bank`) |
| Model chosen by its score on synthetic validation | The best checkpoint and the threshold are chosen by **F1 on real labelled tiles** |
| U-Net trained from scratch on fewer than 50 images | ImageNet-pretrained encoder plus scSE attention |
| BCE loss gives up and predicts no damage when damage covers few pixels | Dice + Focal loss; the threshold is tuned rather than fixed at 0.5 |
| Different scale or preprocessing at test time | One `seg_preprocess` function is shared by training and inference; training uses random scales (0.5–1.5×); inference runs tiled at the working resolution |

## Setup

```bash
# environment (already created at D:\AI_Project\env)
conda create -p ./env python=3.11 -y
env\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
env\python -m pip install -r requirements.txt
env\python scripts/download_weights.py        # big-lama.pt + RealESRGAN_x4plus.pth
```

All commands below assume `env\python` (or activate the env: `conda activate D:\AI_Project\env`).
For the notebook, choose the **"AI Restoration (py3.11)"** kernel. It is already registered.

Note: this network blocks GitHub's release-asset server, so `download_weights.py` falls back to Hugging Face mirrors. The weights are already in `weights/`.

**Smoke test without real data:** `python scripts/make_demo_data.py` creates procedural folk-art-style images in `data/demo/`. Use them only to check that the scripts run, never as training data for the real model.

## Data layout

```
data/clean/                  undamaged or well-preserved paintings, or clean crops of damaged ones
data/damaged/                REAL damaged paintings you want to restore
data/real_labeled/images/    real damaged tiles (created by make_label_tiles.py)
data/real_labeled/masks/     your corrected masks: 0 bg, 1 crack, 2 loss, 3 stain, 255 damage of unknown type
data/damage_bank/            real damage cut-outs (created by build_damage_bank.py)
```

If you have very few clean paintings, crop the undamaged areas out of damaged ones and put the crops in `data/clean/`. Public-domain folk paintings, for example from Wikimedia Commons, also work.

### Current dataset: Kaggle "Indian paintings" (MIT, 8 styles)
```bash
python scripts/import_dataset.py "C:/Users/Atharva/Downloads/archive"   # -> data/pool (862 after removing duplicates and images <400px)
python scripts/split_pool.py                                            # -> data/clean (806), data/damaged (21)
```
- `data/pool/damaged_ids.txt` lists 21 paintings with real visible damage (flaking, abrasion, stains, losses), checked on 1:1 crops.
- `data/pool/exclude_ids.txt` lists aged works whose damage is unclear, plus photos that aren't paintings. They are kept out of `clean`, because any unlabelled damage in the clean pool would teach the model that damage is normal.
- Edit the id lists (ids are shown in `outputs/contact_sheets/`) and rerun `split_pool.py`.
- The real damage is mostly flaking, stains and abrasion rather than fine craquelure, because web JPEGs blur fine cracks.

### Pattachitra and Kalamkari: Hugging Face "Indian Art Styles" (CC BY 4.0), Wikimedia subset only
Source: https://huggingface.co/datasets/Divya0001/indian-art-styles (citation in its card: Pattanayak, 2026). Its labels are **unreliable**:
- The V&A part (85% of both classes) consists of unrelated photographs and drawings, and the same images appear under both labels.
- The Met part consists of Chinese paintings.
- Only the Wikimedia part is usable, and those images were downscaled to 512 px. About half are documentary photos of artists and markets.
```bash
# downloads data/external/hf_wikimedia/{pattachitra,kalamkari} (see data/external/wikimedia_subset.csv)
python scripts/import_dataset.py data/external/hf_wikimedia --append --prefix hf_   # ids 862..1131
python scripts/split_pool.py
```
After hand curation (contact sheets in `outputs/contact_sheets_hf/`), this added 106 clean paintings and 4 damaged ones. The current totals are **912 clean / 25 real-damaged**.

## Workflow

```bash
# 0. Check that synthetic damage looks like your real damage
python scripts/preview_synth.py                     # -> outputs/synth_preview.png

# 1. First model, trained on synthetic data only
python scripts/train_seg.py --epochs 15

# 2. Label real tiles (the most effective single step; 20-40 tiles is enough to begin)
python scripts/make_label_tiles.py --per-image 6    # suggested masks come from the model
python scripts/label_tool.py                        # correct them with the brush, then press s
python scripts/build_damage_bank.py                 # real damage cut-outs for copy-paste

# 3. Retrain with the real labels (selection by real F1)
python scripts/train_seg.py --epochs 40
#    Repeat steps 2-3 once or twice. Each pass, label the tiles where the model is wrong.

# 4. Colour model
python scripts/train_color.py --epochs 30

# 5. Restore
python scripts/restore.py data/damaged --sr-scale 2
python scripts/restore.py painting.jpg --color-ref protected_margin.png   # reference-based colour
```

Each restored painting gets a folder in `outputs/restored/<name>/` containing the mask and overlay, the inpainted image, the colour-corrected image, the final super-resolved image, `confidence.npy` and `confidence_map.png` (green = original or reliable, red = guessed infill), and `report.json`.

## Held-out test set (the professor's Mandawa paintings)

The data is split into three sets:

| Split | Data | Used for |
|---|---|---|
| train | `data/clean` (synthetic damage) + labelled tiles from `data/damaged` + `damage_bank` | fitting weights |
| validation | 30% of the labelled real tiles from `data/damaged` | choosing the checkpoint and threshold |
| **test** | `data/test_holdout/` (Mandawa) | **final evaluation only, run once** |

- No training or labelling script reads `data/test_holdout`. Do not copy these images anywhere else, not even into `damage_bank`.
- Run `env\python scripts\evaluate_test.py` only after development is finished. It first checks for leaks (the same image, even re-encoded, appearing in any training folder), then evaluates with the frozen checkpoint and threshold.
- For F1/IoU numbers, hand-label the test images into `data/test_holdout/masks/` (for example with `label_tool.py --root data/test_holdout`, after moving the images into `images/`). Labelling test images for evaluation is fine; using those labels for training or tuning is not.
- Expect a domain gap: Mandawa frescoes are lime plaster, while the training data is mostly paper and cloth. Report the result honestly.

## Validation notebook

`notebooks/validation.ipynb` contains:
- training loss and F1 curves for segmentation and colour
- how many patches each painting yields
- a comparison of the classical detector and the U-Net on real labelled tiles
- per-class metrics

## How the confidence map is computed (M4)

The confidence of each pixel combines three signals:
1. **Disagreement between LaMa runs.** The image is inpainted again with the input flipped and with the mask enlarged by 4 px and 8 px. Wherever these versions differ, the model is guessing.
2. **Distance from real paint.** The deeper a pixel is inside a missing area, the less evidence it has.
3. **Detection ambiguity.** This is high when the segmentation probability is close to the threshold, or when the test-time-augmented predictions disagree.

The PatchGAN discriminator saved with the colour model (`checkpoints/color_disc_best.pt`) can serve as an additional per-patch realism score.

## Memory (RTX 4060, 8 GB)

- Segmentation training: batch 8 at 384 px with AMP. Lower `--batch` if you run out of memory.
- Inference: every stage runs on overlapping tiles and blends them with a cosine window, so any image size fits and there are no seams. LaMa only processes windows that contain damage.
- Real-ESRGAN runs in fp16 on 256 px tiles.

## Extensions
- **CycleGAN (unpaired)** for de-fading: map truly faded areas to the un-faded margins. It would sit next to `color.py`, reusing `PatchDiscriminator`.
- **Fine-tuning LaMa** on folk-art tiles with synthetic masks.
- **Per-class inpainting:** inpaint cracks and losses only, and fix stains with the colour stage (`--classes crack,loss`).
