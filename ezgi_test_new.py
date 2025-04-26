# test_models.py
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer
import numpy as np
from transformers import TrainerCallback
from datasets import Dataset
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score
from safetensors.torch import load_file
from transformers import BertConfig,BertPreTrainedModel, BertModel





# === Dosya yolları
test_path = Path("test_stratified.csv")
train_path = Path("train_stratified.csv")


# === Etiket listeleri

normal = ["severe_toxicity","obscene","identity_attack","insult","threat"]
imbal  = [
    "asian","atheist","bisexual","black","buddhist","christian","female",
    "heterosexual","hindu","homosexual_gay_or_lesbian","jewish","latino",
    "male","muslim","psychiatric_or_mental_illness","transgender","white"
]

# < 2 pozitifli etiketleri çıkar
rare_drop = [
    "severe_toxicity","other_disability","intellectual_or_learning_disability",
    "other_sexual_orientation","other_gender","other_race_or_ethnicity",
    "other_religion","physical_disability","hindu","buddhist"
]

normal_labels = [l for l in normal if l not in rare_drop]
imbalanced_labels  = [l for l in imbal  if l not in rare_drop]
all_labels = normal_labels + imbalanced_labels

# === Test ve train seti elde et
train = pd.read_csv(train_path)
train = train[["comment_text"] + all_labels].dropna()
train[all_labels] = (train[all_labels] >= .5).astype(int)
print("train set başarıyla okundu...")

class FocalLoss(nn.Module):
    def __init__(self, γ=2.): super().__init__(); self.g=γ
    def forward(self,x,y):
        bce = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        pt  = torch.exp(-bce)
        return ((1-pt)**self.g * bce).mean()

class DualHeadBert(BertPreTrainedModel):
    def __init__(self, config, pos_w_n):
        super().__init__(config)
        self.bert = BertModel(config)
        h = config.hidden_size
        self.head_n = nn.Linear(h, len(normal_labels))
        self.head_i = nn.Linear(h, len(imbalanced_labels))
        self.loss_n = nn.BCEWithLogitsLoss(pos_weight=pos_w_n)
        self.loss_i = FocalLoss()

        self.init_weights()  # Huggingface modelleri için şart!

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                head="n", labels=None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled = outputs.pooler_output

        if head == "n":
            logits = self.head_n(pooled)
            loss = self.loss_n(logits, labels) if labels is not None else None
        else:
            logits = self.head_i(pooled)
            loss = self.loss_i(logits, labels) if labels is not None else None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
# 2) Test seti yükleme
test = pd.read_csv(test_path)
test = test[["comment_text"] + all_labels].dropna()
test[all_labels] = (test[all_labels] >= 0.5).astype(int)
print("test set başarıyla okundu...")

X_test = test["comment_text"].tolist()
y_test = test[all_labels].values.astype(int)
# dilimleme
y_test_n = y_test[:, :len(normal_labels)]
y_test_i = y_test[:, len(normal_labels):]



# 3) Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_n = BertConfig.from_pretrained("saved_model_n")
model_n = DualHeadBert.from_pretrained("saved_model_n", config=config_n, pos_w_n=pos_weight_n).eval()
model_n.to(device)
print("Model_n başarıyla okundu...")

config_i = BertConfig.from_pretrained("saved_model_i")
model_i = DualHeadBert.from_pretrained("saved_model_i", config=config, pos_w_n=pos_weight_n).eval()
model_i.to(device)
print("Model_i başarıyla okundu...")



# 4) Tokenizer'ı da yükle
tokenizer = BertTokenizer.from_pretrained("saved_tokenizer")



# 5) Toplu tahmin fonksiyonu (batch processing with progress)
def batch_predict(model, texts, head, bs=32):
    all_logits = []
    total = len(texts)
    print(f"Starting batch prediction: total samples = {total}, batch size = {bs}")
    for start_idx in range(0, total, bs):
        end_idx = min(start_idx + bs, total)
        batch = texts[start_idx:end_idx]
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs)["logits"]
        all_logits.append(logits.cpu().numpy())
        print(f"Processed {end_idx}/{total} samples")
    return np.vstack(all_logits)

# 6) Tahmin
print("\n=== Normal model prediction ===")
logits_n = batch_predict(model_n, X_test,"n")
print("\n=== Imbalanced model prediction ===")
logits_i = batch_predict(model_i, X_test,"i")

# 7) Olasılıklar ve tahminler
probs_n = 1 / (1 + np.exp(-logits_n))
probs_i = 1 / (1 + np.exp(-logits_i))
preds_n = (probs_n > 0.5).astype(int)
preds_i = (probs_i > 0.5).astype(int)

print("preds_i shape:", preds_i.shape)
print("y_test_i shape:", y_test_i.shape)
print("imbalanced_labels:", len(imbalanced_labels))
print("Model head_i out features:", model_i.head_i.out_features)


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curves(y_true, probs, label_names, title):
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap = average_precision_score(y_true[:, i], probs[:, i])
        plt.plot(recall, precision, label=f"{label} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({title})")
    plt.legend(loc="lower left", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("pr_curve_normal.png", dpi=300)

# Normal-label model için
plot_pr_curves(y_test_n, probs_n, normal_labels, "Normal Labels")

# Imbalanced-label model için
plot_pr_curves(y_test_i, probs_i, imbalanced_labels, "Imbalanced Labels")

# === Farklı gamma değerleri için test döngüsü
gamma_list = [0.5, 1.0, 2.0, 3.0]
alpha_list = [0.25, 0.5, 0.75, 1.0]

results = []



# 8) Normal etiketler için metrikler
print("\n=== Normal-label model metrics ===")
for idx, lbl in enumerate(normal_labels):
    acc = accuracy_score(y_test_n[:, idx], preds_n[:, idx])
    p, r, f1, _ = precision_recall_fscore_support(
        y_test_n[:, idx], preds_n[:, idx], average="binary", zero_division=0
    )
    print(f"{lbl:25s}  ACC={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

# 9) Imbalanced etiketler için metrikler
print("\n=== Imbalanced-label model metrics ===")
for idx, lbl in enumerate(imbalanced_labels):
    acc = accuracy_score(y_test_i[:, idx], preds_i[:, idx])
    p, r, f1, _ = precision_recall_fscore_support(
        y_test_i[:, idx], preds_i[:, idx], average="binary", zero_division=0
    )
    print(f"{lbl:25s}  ACC={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

# 10) Genel ortalama
all_preds = np.concatenate([preds_n, preds_i], axis=1)
all_acc = np.mean([accuracy_score(y_test[:, i], all_preds[:, i])
                   for i in range(y_test.shape[1])])
print(f"\n🔥 Overall Average Accuracy: {all_acc:.4f}")