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

```python src/run_experiment.py```

