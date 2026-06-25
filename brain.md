# 🧠 PBCAT-M Brain (Future Work & Backlog)

This file serves as a memory bank and backlog for tasks that we have intentionally deferred to a later stage of the project.

## Deferred Tasks

### 1. Data Labeling & NOAA Matcher
* **Context**: We need to map our 6-hour SoLEXS/HEL1OS FITS windows to actual solar flare labels (0=None, 1=C, 2=M, 3=X) so the model has "ground truth" answers to train on.
* **Why Deferred**: We decided to focus on building the complete forward pass architecture (Stages 1 through 7) first. Labeling is only strictly required once we write the `train.py` loop.
* **Action Required**: 
    - Determine if the flare catalog will be provided directly by ISRO (custom CSV) or if we will use the standard NOAA GOES XRS JSON/CSV catalogs.
    - Build `noaa_matcher.py` to parse this catalog and assign target labels based on timestamps.
    - Create a standard PyTorch `Dataset` class to wrap `stage1/loader.py` and the matcher.

### 2. PyTorch Training Loop
* **Context**: The `train.py` skeleton currently exists but needs to be fleshed out with the actual Dataset, custom loss functions (OOD detection constraints), and hardware scaling.
* **Why Deferred**: We need the model architecture built first.
* **Action Required**:
    - Implement the custom evaluation metrics: TSS (True Skill Statistic), HSS (Heidke Skill Score), and ECE (Expected Calibration Error).
    - Setup Mixed Precision Training (`torch.cuda.amp.GradScaler`) for the A100/V100 hardware.

### 3. ONNX Deployment Export (Stage 7+)
* **Context**: The model ultimately needs to run in inference mode at the ISRO Space Weather Monitoring Center.
* **Why Deferred**: Model needs to be trained and validated first.
* **Action Required**: 
    - Add an export script to convert the final PyTorch weights to `.onnx`.
    - Ensure all custom Mamba/PyTorch layers are ONNX-compatible.
