import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DeeBERTモデル定義（中間層分類器を持つBERT）
# ---------------------------------------------------------
class DeeBertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.num_layers = self.bert.config.num_hidden_layers
        self.num_labels = num_labels
        
        # 各Transformerブロック（中間層）の直後に配置する軽量分類器 (Off-ramps)
        hidden_size = self.bert.config.hidden_size
        self.off_ramps = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels)
            ) for _ in range(self.num_layers)
        ])

    def forward(self, input_ids, attention_mask, eval_threshold=None):
        """
        eval_threshold が None の場合: 通常学習（全中間層の損失を計算）
        eval_threshold が float の場合: 動的推論（エントロピーが閾値未満で打ち切り）
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:] # Embedding層を除いた各層の出力 [12, Batch, Seq, Hidden]

        if eval_threshold is None:
            # 【学習時】各中間層の logits を計算
            all_logits = []
            for layer_idx in range(self.num_layers):
                # [CLS] トークンのベクトルを各中間分類器に入力
                cls_rep = hidden_states[layer_idx][:, 0, :]
                logits = self.off_ramps[layer_idx](cls_rep)
                all_logits.append(logits)
            return all_logits
        else:
            # 【推論時】エントロピーに基づいた早期打ち切り
            batch_size = input_ids.size(0)
            final_preds = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
            exit_layers = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)
            active_indices = torch.arange(batch_size, device=input_ids.device)

            for layer_idx in range(self.num_layers):
                cls_rep = hidden_states[layer_idx][active_indices, 0, :]
                logits = self.off_ramps[layer_idx](cls_rep)
                probs = torch.softmax(logits, dim=-1)
                
                # エントロピー計算: H(x) = - sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                preds = torch.argmax(logits, dim=-1)

                # 閾値以下の判定（十分な確信度）または 最終層に到達した場合
                exited_mask = (entropy < eval_threshold) | (layer_idx == self.num_layers - 1)
                
                # 打ち切られたサンプルの結果を記録
                exited_indices = active_indices[exited_mask]
                if len(exited_indices) > 0:
                    final_preds[exited_indices] = preds[exited_mask]
                    exit_layers[exited_indices] = layer_idx + 1 # 1-indexed

                # まだ打ち切られていないサンプルのみ次の層へ
                active_indices = active_indices[~exited_mask]
                if len(active_indices) == 0:
                    break

            return final_preds, exit_layers

# ---------------------------------------------------------
# 2. データ準備 (JGLUE / MARC-ja: 2クラス感情分析)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Parquet形式で配られている最新互換リポジトリから取得
# (MARC-ja タスクを指定)
raw_dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

# データセットの列名を確認して前処理
# ※ MARC-ja のテキスト列は "sentence" または "text"、ラベルは "label"
def preprocess_function(examples):
    text_column = "sentence" if "sentence" in examples else "text"
    return tokenizer(
        examples[text_column], 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )

# トークナイズ処理とPyTorch DataLoader化
tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 学習・検証用の分割データ取得
train_loader = DataLoader(tokenized_dataset["train"].shuffle(seed=42).select(range(3000)), batch_size=32)
val_loader = DataLoader(tokenized_dataset["validation"].select(range(1000)), batch_size=32)

# ---------------------------------------------------------
# 3. 2段階学習 (Two-Stage Fine-Tuning)
# ---------------------------------------------------------
model = DeeBertForSequenceClassification(model_name, num_labels=2).to(device)

# --- Stage 1: Backbone + 最終層の学習 ---
optimizer_st1 = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

model.train()
print("=== Stage 1: Backbone & Final Layer Fine-tuning ===")
for epoch in range(1): # デモ用に1エポック
    for batch in train_loader:
        optimizer_st1.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        all_logits = model(input_ids, attention_mask)
        loss = criterion(all_logits[-1], labels) # 最終層のみの損失
        loss.backward()
        optimizer_st1.step()

# --- Stage 2: Backboneをフリーズし、中間分類器（Off-ramps）を学習 ---
for param in model.bert.parameters():
    param.requires_grad = False

optimizer_st2 = optim.AdamW(model.off_ramps.parameters(), lr=1e-3)

print("=== Stage 2: Off-ramps Fine-tuning ===")
model.train()
for epoch in range(1):
    for batch in train_loader:
        optimizer_st2.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        all_logits = model(input_ids, attention_mask)
        # 全層の損失の総和（あるいは平均）をとって最適化
        total_loss = sum(criterion(logits, labels) for logits in all_logits)
        total_loss.backward()
        optimizer_st2.step()

# ---------------------------------------------------------
# 4. 評価実験（エントロピー閾値を変えて精度と退出位置を計測）
# ---------------------------------------------------------
model.eval()

thresholds = [0.0, 0.1, 0.3, 0.5, 0.7] # 0.0は事実上全層パス
results = {}

with torch.no_grad():
    for thr in thresholds:
        all_preds = []
        all_exits = []
        all_labels = []

        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds, exit_layers = model(input_ids, attention_mask, eval_threshold=thr)
            
            all_preds.extend(preds.cpu().numpy())
            all_exits.extend(exit_layers.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_exit_layer = np.mean(all_exits)
        
        results[thr] = {
            "accuracy": accuracy,
            "avg_layer": avg_exit_layer,
            "exit_distribution": all_exits
        }
        print(f"Threshold: {thr:.1f} | Accuracy: {accuracy*100:.2f}% | Avg Exit Layer: {avg_exit_layer:.2f}")

# ---------------------------------------------------------
# 5. 結果の可視化
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# (1) 平均打ち切り層数 vs 正解率 (Trade-off Curve)
avg_layers = [res["avg_layer"] for res in results.values()]
accuracies = [res["accuracy"] * 100 for res in results.values()]

ax1.plot(avg_layers, accuracies, marker='o', color='b', linestyle='-')
ax1.set_xlabel("Average Exit Layer Number")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Trade-off between Speed (Exit Layer) and Accuracy")
ax1.grid(True)

# (2) 特定閾値における打ち切り位置の分布ヒストグラム
target_thr = 0.3
ax2.hist(results[target_thr]["exit_distribution"], bins=range(1, 14), align='left', rwidth=0.8, color='orange')
ax2.set_xticks(range(1, 13))
ax2.set_xlabel("Exit Layer")
ax2.set_ylabel("Sample Count")
ax2.set_title(f"Exit Layer Distribution (Threshold = {target_thr})")

plt.tight_layout()
plt.show()