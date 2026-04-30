# transfer-learning

## Install torch with gpu

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

## Correct Kaggle Dataset Download Flow
Step 1: Install Kaggle API
``` bash
pip install kaggle
```
---
Step 2: Authenticate 

- Go to Kaggle → Account → API

- Generate legacy API Token

- You receive `kaggle.json` 

- Place token in:

    ```C:\Users\<USER>\.kaggle\kaggle.json```
    
(create .kaggle if it does not exist)

---
Step 3: Download dataset
``` bash
python src/get_kaggle_data.py
```
---
Step 4. Correct Data Structure
Move downloaded file to data package

After extraction:

    data/corn
     ├── train/
     ├── test/
     ├── train.csv
     ├── test.csv
     ├── sample_submission.csv
---

## Start training

```bash
# New run
python run_experiment.py --note "augmentation v2"

# Resume after crash
python run_experiment.py --run-id 2025-04-30_14-22-01
```

Run ID is printed at startup and matches the filename in `results/`.

### Gotchas

- Resume skips `done` experiments and retries `failed` and `running` ones — each retried experiment always starts from epoch 1, not from where it was interrupted.
- Forgetting `--run-id` after a crash creates a new file — old progress is not lost but not resumed either.
- `--run-id` must match the filename exactly: `YYYY-MM-DD_HH-MM-SS`.

