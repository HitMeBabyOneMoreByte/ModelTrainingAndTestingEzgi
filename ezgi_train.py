# pip install iterstrat
# FINAL CODE  ─────────────────────────────────────────────────────────────
import torch, torch.nn as nn
from transformers import TrainerCallback
import transformers
import pandas as pd, numpy as np
from datasets import Dataset
from transformers import (BertTokenizer, TrainingArguments, Trainer)
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score

#──────────────────────────────────────────────────────────────────────────
# 1)  VERİ ve ETİKET LİSTELERİ
#──────────────────────────────────────────────────────────────────────────
df = pd.read_csv("train.csv")

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

normal = [l for l in normal if l not in rare_drop]
imbal  = [l for l in imbal  if l not in rare_drop]
all_lbl = normal + imbal

df = df[["comment_text"] + all_lbl].dropna()
df[all_lbl] = (df[all_lbl] >= .5).astype(int)

X = df["comment_text"].values.reshape(-1,1)
Y = df[all_lbl].values.astype(np.float32)

#──────────────────────────────────────────────────────────────────────────
# 2)  STRATIFIED 80 / 15 / 5 BÖLÜNÜM
#──────────────────────────────────────────────────────────────────────────
X_tv, Y_tv, X_test, Y_test = iterative_train_test_split(X, Y, test_size=.05)
X_train, Y_train, X_val,  Y_val = iterative_train_test_split(
    X_tv, Y_tv, test_size=.1579)                 # 15 %

def to_df(x, y):
    d = pd.DataFrame({"comment_text": x.flatten()})
    d[all_lbl] = y
    return d

train, val, test = map(to_df, (X_train, X_val, X_test), (Y_train, Y_val, Y_test))
print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

#──────────────────────────────────────────────────────────────────────────
# 3)  pos_weight (head‑N)
#──────────────────────────────────────────────────────────────────────────
pos_n = train[normal].sum(0).values
neg_n = len(train) - pos_n
pos_weight_n = torch.tensor(neg_n / (pos_n + 1e-8), dtype=torch.float32)
print("pos_weight‑N:", np.round(pos_weight_n.numpy(),2).tolist())
#──────────────────────────────────────────────────────────────────────────
# 4) CALLBACKS: Gradient Clipping ve NaN Kontrolü
#──────────────────────────────────────────────────────────────────────────

class GradientClippingCallback(TrainerCallback):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class GradCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradients of {name}")


#──────────────────────────────────────────────────────────────────────────
# 4)  TOKENIZER & DATASET
#──────────────────────────────────────────────────────────────────────────
tok = BertTokenizer.from_pretrained("bert-base-uncased")
def tok_ds(frame, cols):
    enc = tok(frame["comment_text"].tolist(),
              padding="max_length", truncation=True, max_length=128)
    return Dataset.from_dict({**enc, "labels": frame[cols].values.astype(np.float32)})

ds_tr_n, ds_va_n, ds_te_n = map(lambda d: tok_ds(d, normal), (train,val,test))
ds_tr_i, ds_va_i, ds_te_i = map(lambda d: tok_ds(d, imbal ), (train,val,test))

#──────────────────────────────────────────────────────────────────────────
# 5)  LOSS FONKSİYONLARI
#──────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, γ=2.): super().__init__(); self.g=γ
    def forward(self,x,y):
        bce = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        pt  = torch.exp(-bce)
        return ((1-pt)**self.g * bce).mean()

#──────────────────────────────────────────────────────────────────────────
# 6)  TEK BERT + ÇİFT HEAD MODELİ
#──────────────────────────────────────────────────────────────────────────
class DualHeadBert(nn.Module):
    def __init__(self, pos_w_n):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        h = self.bert.config.hidden_size
        self.head_n = nn.Linear(h, len(normal))
        self.head_i = nn.Linear(h, len(imbal))
        self.loss_n = nn.BCEWithLogitsLoss(pos_weight=pos_w_n)
        self.loss_i = FocalLoss()

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, head="n", labels=None):
        pooled = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids).pooler_output
        if head == "n":
            logits = self.head_n(pooled)
            loss = self.loss_n(logits, labels) if labels is not None else None
        else:
            logits = self.head_i(pooled)
            loss = self.loss_i(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

model_n = DualHeadBert(pos_weight_n)
model_i = DualHeadBert(pos_weight_n)

#──────────────────────────────────────────────────────────────────────────
# 7)  ORTAK TRAINING ARGUMENTS  (grad‑clipping & stab. ayarlar)
#──────────────────────────────────────────────────────────────────────────
common_args = dict(
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,      # efektif batch 16
    num_train_epochs            = 6,
    learning_rate               = 2e-5,
    warmup_ratio                = .1,
    lr_scheduler_type           = "cosine",
    weight_decay                = .01,
    fp16                        = torch.cuda.is_available(),
    max_grad_norm               = 1.0,    # ← clipping
    logging_steps               = 500,
    eval_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    remove_unused_columns       = False,
    metric_for_best_model       = "eval_f1_macro",
    greater_is_better           = True,
    report_to                   = "none",
)

#──────────────────────────────────────────────────────────────────────────
# 8)  TRAINER YARDIMCISI
#──────────────────────────────────────────────────────────────────────────
def make_trainer(flag, dst, dsv, model):
    def metr(p):
        logits, lab = p
        preds = (torch.sigmoid(torch.tensor(logits))>.5).int().numpy()
        return {"eval_f1_macro": f1_score(lab.astype(int), preds,
                                          average='macro', zero_division=0)}
    return Trainer(
        model=model,
        args = TrainingArguments(output_dir=f"out_{flag}", **common_args),
        train_dataset=dst, eval_dataset=dsv,
        compute_metrics=metr,
        data_collator=lambda b:{
            "input_ids": torch.tensor([i["input_ids"] for i in b]),
            "attention_mask": torch.tensor([i["attention_mask"] for i in b]),
            "token_type_ids": torch.tensor([i["token_type_ids"] for i in b]),
            "labels": torch.tensor([i["labels"] for i in b]),
            "head": flag
        },
        callbacks=[GradientClippingCallback(), GradCheckCallback()]  # ← eklendi
    )


tr_n = make_trainer("n", ds_tr_n, ds_va_n, model_n)
tr_i = make_trainer("i", ds_tr_i, ds_va_i, model_i)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_n.to(device)
model_i.to(device)

print("▶ Head‑N (threat / insult / obscene / id‑attack)"); tr_n.train()
print("▶ Head‑I (identity labels)");                      tr_i.train()

#──────────────────────────────────────────────────────────────────────────
# 9)  VALIDATION TABANLI EŞİKLER
#──────────────────────────────────────────────────────────────────────────
def best_thresholds(tr, ds):
    out = tr.predict(ds)
    prob, lab = torch.sigmoid(torch.tensor(out.predictions)).numpy(), out.label_ids
    th = []
    for i in range(lab.shape[1]):
        best, bt = 0, .5
        for t in np.arange(.1,.9,.02):
            f1 = f1_score(lab[:,i], (prob[:,i]>t).astype(int), zero_division=0)
            if f1>best: best, bt = f1, t
        th.append(bt)
    return np.array(th)

th_n = best_thresholds(tr_n, ds_va_n)
th_i = best_thresholds(tr_i, ds_va_i)

#──────────────────────────────────────────────────────────────────────────
# 10) TEST SONUCU
#──────────────────────────────────────────────────────────────────────────
def test_f1(tr, ds, th):
    out = tr.predict(ds)
    preds = (torch.sigmoid(torch.tensor(out.predictions)).numpy()>th).astype(int)
    return f1_score(out.label_ids, preds, average='macro', zero_division=0)

print("\n=== TEST RESULTS ===")
print("Normal    macro‑F1 :", test_f1(tr_n, ds_te_n, th_n))
print("Identity  macro‑F1 :", test_f1(tr_i, ds_te_i, th_i))


tr_n.model.save_pretrained("saved_model_n")
tr_i.model.save_pretrained("saved_model_i")
tok.save_pretrained("saved_tokenizer")

