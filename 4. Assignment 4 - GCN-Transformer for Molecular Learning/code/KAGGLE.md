# Running this project on Kaggle (GPU)

The project is self-contained: `Lipophilicity.csv` is bundled and `config.py`
resolves it by a relative path, so no path edits are needed. The full run
(15 warm-up + up to 40 joint epochs, 300 samples per temperature) takes only a few minutes on a
Kaggle GPU.

## One-time account setup
1. Create a free account at https://www.kaggle.com and sign in.
2. Verify your phone number: profile menu -> Settings -> Phone verification.
   Kaggle requires a verified account before it will enable the GPU or Internet.

## Step 1 — Upload the project as a Dataset
1. Left sidebar -> **Create** -> **Dataset** (or **Datasets** -> **New Dataset**).
2. Upload `nndl_project4.zip`. Kaggle automatically extracts the archive.
3. Give it a title (e.g. `nndl-project4`) and click **Create**.

## Step 2 — Create a Notebook and attach the data
1. Left sidebar -> **Create** -> **Notebook**.
2. In the right-hand panel click **Add Input** and add the dataset you just made.
   It will appear under `/kaggle/input/<your-dataset-name>/`.

## Step 3 — Enable GPU and Internet
In the right-hand **Settings / Session options** panel:
- **Accelerator** -> **GPU** (P100 or T4 is fine).
- **Internet** -> **On** (needed for `pip install`).
The session restarts when you change the accelerator.

## Step 4 — Run these cells

Cell 1 — confirm the GPU is attached:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Cell 2 — install the extra packages (torch is already present on Kaggle):
```python
!pip install -q torch_geometric rdkit selfies
```

Cell 3 — copy the project into the writable working directory and run it:
```python
import os, shutil

# locate the uploaded project (folder that contains main.py + config.py)
src = None
for root, _, files in os.walk("/kaggle/input"):
    if "main.py" in files and "config.py" in files:
        src = root
        break
assert src, "Project not found under /kaggle/input — is the dataset attached?"

proj = "/kaggle/working/nndl_project4"
if os.path.exists(proj):
    shutil.rmtree(proj)
shutil.copytree(src, proj)
os.chdir(proj)
print("Running from:", proj)
```
```python
!python main.py
```

Cell 4 — zip the outputs so you can download them:
```python
import shutil
shutil.make_archive("/kaggle/working/nndl_outputs", "zip",
                    "/kaggle/working/nndl_project4/output")
print("Done. Download nndl_outputs.zip from the right-side Output panel.")
```

## Step 5 — Download and share
In the right-hand **Output** panel (or the `/kaggle/working` file browser), download
`nndl_outputs.zip`. It contains `output/section1/` and `output/section2/` with all
statistics, curves, generation metrics, `results.json`, the training log, and the
accepted-novel molecule grid.

## Notes
- To change how long it trains, edit the top of `config.py` or override inline, e.g.
  `!LM_EPOCHS=15 JOINT_EPOCHS=40 N_GENERATE=300 python main.py`.
- If a cell reports the GPU is not available, re-check that the accelerator is set to
  GPU and that your account is phone-verified, then restart the session.
- The code auto-detects the GPU (`config.DEVICE`); no code changes are required to use it.
