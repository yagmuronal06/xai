# xai_final_assignment

Comparing Grad-CAM and Score-CAM on the Stanford Dogs Dataset for my XAI course assignment.

The main question: does a gradient-free method (Score-CAM) do a better job of highlighting the actual dog features compared to the gradient-based Grad-CAM, especially when the model misclassifies similar-looking breeds?

---

## Files

```
├── model.py           # ResNet-50 fine-tuning
├── explainability.py  # Grad-CAM and Score-CAM
├── evaluation.py      # deletion/insertion AUC, pointing game
├── demo.py            # runs everything end to end
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/yagmuronal06/xai_final_assignment
cd xai_final_assignment
pip install -r requirements.txt
```

---

## Dataset

Download Stanford Dogs from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) and extract it so the structure looks like:

```
stanford-dogs/
└── Images/
    ├── n02085620-Chihuahua/
    ├── n02085782-Japanese_spaniel/
    └── ...
```

---

## Usage

```bash
python demo.py --data_dir /path/to/stanford-dogs --mode both
```

---
