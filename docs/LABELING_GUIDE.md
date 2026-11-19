# Damage Labelling Guide

You are marking **damage** on small square tiles cut from real paintings and frescoes. The model learns from these masks to find damage by itself, so labels must be **consistent across all four of us**. Please read this guide fully once before you start.

## Setup (once)
1. Install Python 3.10 or newer from python.org. Tick "Add Python to PATH" during installation.
2. Unzip your kit anywhere.
3. Double-click **`run_label.bat`**. The first run installs `opencv-python` and `numpy` automatically.
   - Mac/Linux: `pip3 install opencv-python numpy`, then `python3 scripts/label_tool.py`

A window opens with your first tile. Tiles you have saved are skipped the next time you start, so you can stop and resume whenever you like.

## Controls
| Action | Control |
|---|---|
| Paint damage | left-click and drag |
| Erase | right-click and drag |
| **Zoom in/out at the cursor** | **mouse wheel** (or `+` / `-`) |
| Pan while zoomed | hold the wheel button and drag, or use the arrow keys |
| Reset the zoom | `0` |
| Brush size | `[` smaller, `]` bigger. The circle around the cursor shows the brush. |
| Damage type | `1` `2` `3` `4` (see below) |
| Hide or show your paint | `h` |
| Clear the tile / reset to the suggestion | `c` / `r` |
| **Save and go to the next tile** | **`s`** |
| Skip / previous tile | `n` / `p` |
| Quit | `q` or `Esc` |

Some tiles start with a suggested mask from the current model. These suggestions are often wrong or incomplete. Fix them; don't trust them.

## What to label: ask "is the original paint still there?"

| Key | Type | Paint it on |
|---|---|---|
| `1` red | **Crack** | Thin lines where the paint or paper has split |
| `2` blue | **Loss**: the paint is gone | Flaked spots, tiny bare white flecks, tears, holes, rubbed-through areas where the base shows. On frescoes: fallen plaster, exposed brick or lime. |
| `3` yellow | **Stain**: the paint is still there but covered or discoloured | Brown water marks or tide-lines, foxing spots, mold, dirt. On frescoes: white salt haze, soot, water streaks. |
| `4` magenta | **Damage of unknown type** | Clearly damage, but you can't tell which type |

Cases that look alike:
- White specks with sharp edges and flat colour are **2**. A cloudy white veil with the painting visible through it is **3**.
- A dark hole or tear is **2**. A brown ring or spot on intact paper is **3**.

## Do NOT paint
- **Painted lines, outlines, hatching or dot patterns**, even when they are thin and dark. Learning this difference is the main point of the exercise.
- General yellowing or fading of old paper or walls. The colour stage handles that.
- Frames, mounts, background, or wall areas outside the painting.

## Rules of thumb (so we all label the same way)
- **Be rough and slightly generous.** Cover the damage fully plus 2–3 pixels around it. Pixel-perfect edges are not needed.
- **Unsure whether something is damage at all? Leave it unpainted.**
- **Unsure of the type? Use `4`.**
- **A tile with no damage:** press `c` then `s`. Empty tiles are useful examples too.
- Zoom in on areas with small flecks or fine cracks.
- Budget roughly 30–60 seconds per tile.

## Calibration tiles
The first few tiles (named `calib_...`) are the same for everyone. Label them carefully and independently, without comparing with each other first. We use them to measure how consistently we label. That measurement goes in the report.

## When you're done
Double-click **`return_labels.bat`**. It creates `<your name>_labels.zip` next to the kit. Send that zip back (it contains only your masks).
