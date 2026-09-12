import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from maze_pool import MazePool

# TransformerEncoderLayer の fast path（融合カーネル）による形状バグを回避
torch.backends.mha.set_fastpath_enabled(False)


# ================================================================
# 1. 安全なモデルロード関数
# ================================================================
def load_model_safely(model, path, device="cpu"):
    try:
        checkpoint_state = torch.load(path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]

    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for name, param in checkpoint_state.items():
        if name in current_state:
            if param.shape == current_state[name].shape:
                filtered_state[name] = param
            else:
                skipped_keys.append(f"{name} (Shape Mismatch: saved {tuple(param.shape)} vs model {tuple(current_state[name].shape)})")
        else:
            skipped_keys.append(f"{name} (Key Not Found in Model)")

    model.load_state_dict(filtered_state, strict=False)
    print(f"Successfully loaded {len(filtered_state)} / {len(current_state)} layers.")
    if skipped_keys:
        print("Skipped layers due to mismatch or missing:")
        for key in skipped_keys:
            print(f" - {key}")


# ================================================================
# 1.5. 行動マスキング（壁・盤外に向かう行動を無効化する）
# ================================================================
def compute_valid_action_mask(x, wall_channel_idx=0, agent_channel_idx=3):
    """
    観測テンソル x: (B, C, H, W) から、現在位置で選択可能な行動(上下左右)の
    マスクを計算する。有効 = 盤内 かつ 壁でない。

    これがないと、方策は壁や盤外に向かう行動もロジット上は選び得るため、
    遠いゴールに向かう長い正解シーケンスほど「1手でも間違えると失敗」の
    確率が積み重なってしまう（近い迷路は解けるが遠い迷路は解けない、
    という症状の主因になりやすい）。

    戻り値: (B, 4) の bool テンソル (True=選択可能)
    action定義は MazeEnv.step に合わせる: 0=上 1=下 2=左 3=右
    """
    B, C, H, W = x.shape
    device = x.device
    wall_map = x[:, wall_channel_idx]          # (B, H, W)
    agent_map = x[:, agent_channel_idx]        # (B, H, W)
    agent_flat = agent_map.reshape(B, -1).argmax(dim=1)  # (B,)
    agent_r = agent_flat // W
    agent_c = agent_flat % W

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid = torch.zeros(B, 4, dtype=torch.bool, device=device)
    batch_idx = torch.arange(B, device=device)

    for a, (dr, dc) in enumerate(deltas):
        nr = agent_r + dr
        nc = agent_c + dc
        in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        nr_clamped = nr.clamp(0, H - 1)
        nc_clamped = nc.clamp(0, W - 1)
        is_wall = wall_map[batch_idx, nr_clamped, nc_clamped] > 0.5
        valid[:, a] = in_bounds & (~is_wall)

    return valid


def apply_action_mask(logits, valid_mask, mask_value=-1e9):
    """
    valid_mask=Falseの行動のロジットを非常に小さい値にして、実質選ばれなく
    する。全行動が無効（迷路生成のバグ等で完全に孤立したマス）という
    異常系のときは、NaN化を避けるためマスクをかけない（フォールバック）。
    """
    all_invalid = ~valid_mask.any(dim=1)
    safe_mask = valid_mask.clone()
    if all_invalid.any():
        safe_mask[all_invalid] = True
    return logits.masked_fill(~safe_mask, mask_value)


class PPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

# ================================================================
# 1.6. 2D Rotary Position Embedding (2D RoPE)
# ================================================================
# nn.TransformerEncoderLayer は RoPE を標準サポートしないため、
# Q/K にその場で回転を適用する自前のAttention層を用意する。
#
# 設計:
#   - head_dim を「行(row)用の前半」「列(col)用の後半」に2分割し、
#     それぞれに独立した1D RoPEを適用する（2D RoPE）。
#   - head_dim は 4 の倍数である必要がある
#     （行/列で2分割 → さらにRoPEのペア回転で2分割、のため）。
#   - CLSトークンは座標を持たないため、回転角0（＝回転なし）として扱う。
#     これにより CLS-グリッド間の内積は各グリッドトークンの絶対位置に
#     応じた値になり、CLSは「特別な原点」として振る舞う
#     （ViT系のRoPE実装で一般的な慣習）。
#
# 従来の学習可能な絶対位置埋め込み(row_embedding/col_embedding)は、
# 25マス個別にゼロから「隣に壁があればどうする」という関係を学習し
# 直す必要があった。RoPEは相対位置関係をAttentionの内積計算そのものに
# 組み込むため、同じ関係性（例:「1マス右に壁がある」）をマス間で
# 共有して学習でき、サンプルが少ないレアな配置（ゴールが壁で
# 囲まれているケースなど）でのデータ効率が上がることを期待している。

def _rotate_half(x):
    """最後の次元を半分に割り、(-後半, 前半) の順に結合する（RoPEの基本演算）"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_2d(x):
    """
    head_dim全体を行用の前半/列用の後半に分け、それぞれの半分の中で
    独立に _rotate_half を適用する（行と列の回転が混ざらないようにする）
    """
    d = x.shape[-1]
    half = d // 2
    row_part = x[..., :half]
    col_part = x[..., half:]
    return torch.cat((_rotate_half(row_part), _rotate_half(col_part)), dim=-1)


def _apply_rope_2d(x, cos, sin):
    """
    x: (..., N, head_dim)
    cos, sin: (N, head_dim) をブロードキャスト可能な形に整えたもの
    """
    return x * cos + _rotate_half_2d(x) * sin


def build_2d_rope_tables(grid_size, head_dim, base=10000.0):
    """
    5x5などのグリッド上の各マス（行主走査順）とCLSトークン(先頭)を合わせた
    (num_tokens+1, head_dim) の cos/sin テーブルを事前計算する。
    """
    assert head_dim % 4 == 0, (
        "2D RoPEを使うには head_dim (=d_model // nhead) が4の倍数である必要があります"
    )
    half = head_dim // 2   # 行用/列用それぞれの次元数
    quarter = half // 2    # 軸ごとの周波数ペア数

    inv_freq = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
    positions = torch.arange(grid_size, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (grid_size, quarter)
    emb = torch.cat([freqs, freqs], dim=-1)            # (grid_size, half)
    axis_cos = emb.cos()  # (grid_size, half)  行・列共通の周波数テーブル
    axis_sin = emb.sin()

    rows, cols = [], []
    for r in range(grid_size):
        for c in range(grid_size):
            rows.append(r)
            cols.append(c)
    row_idx = torch.tensor(rows, dtype=torch.long)
    col_idx = torch.tensor(cols, dtype=torch.long)

    grid_cos = torch.cat([axis_cos[row_idx], axis_cos[col_idx]], dim=-1)  # (N, head_dim)
    grid_sin = torch.cat([axis_sin[row_idx], axis_sin[col_idx]], dim=-1)

    # CLSトークン(先頭)は回転なし: cos=1, sin=0
    cls_cos = torch.ones(1, head_dim)
    cls_sin = torch.zeros(1, head_dim)

    full_cos = torch.cat([cls_cos, grid_cos], dim=0)  # (N+1, head_dim)
    full_sin = torch.cat([cls_sin, grid_sin], dim=0)
    return full_cos, full_sin


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Q/KにRoPEを適用するマルチヘッド自己注意"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model は nhead で割り切れる必要があります"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        # x: (B, N, d_model)
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # (B,H,N,Dh)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        cos = rope_cos.unsqueeze(0).unsqueeze(0)  # (1,1,N,Dh) -> ブロードキャスト
        sin = rope_sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rope_2d(q, cos, sin)
        k = _apply_rope_2d(k, cos, sin)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B,H,N,Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(attn_out)


class RoPEEncoderLayer(nn.Module):
    """norm_first構成のTransformerEncoderLayer相当（Attention部分をRoPE版に置換）"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionRoPE(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        h = self.norm1(x)
        attn_out = self.self_attn(h, rope_cos, rope_sin, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        h2 = self.norm2(x)
        ff = self.linear2(self.dropout(self.activation(self.linear1(h2))))
        x = x + self.dropout2(ff)
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1,
                 share_weights=False, use_step_embedding=None,
                 num_experts=1, moe_routing="fixed"):
        """
        share_weights=True にすると、num_layers個の独立した層ではなく、
        「1つの層を重み共有したままnum_layers回繰り返し適用する」
        Universal Transformer的な構成になる。

        これは Value Iteration Network (VIN) が「同じ畳み込みカーネルを
        K回繰り返し適用する」ことで、グリッド上のどこでも同じ物理法則
        （隣の価値を見て自分の価値を更新する）が成り立つという帰納バイアスを
        表現しているのと同じ発想。壁マスク(use_wall_mask=True)と組み合わせ、
        num_layersを迷路の直径程度まで増やすことで、Attentionを
        「1層=1ホップの価値伝播」として機能させることを狙っている。

        追加: use_step_embedding
        share_weights=Trueの場合、モデルは「今が何回目の反復か」を知る
        手がかりが無く、1回目(初期特徴抽出)と8回目(伝播完了)で同じ変換を
        強制されてしまう問題が指摘された(Dehghani et al., Universal
        Transformer論文と同じ課題)。これを緩和するため、反復ごとに異なる
        学習可能なベクトル(step embedding)をトークン特徴量に加算できる
        ようにした。既定はNone=share_weightsと同じ値
        （重み共有時のみ自動でON、独立層の場合はOFF）。

        追加: num_experts / moe_routing
        「完全な重み共有(1個)」と「完全な独立層(num_layers個)」の中間として、
        num_experts個(例:4個)の重みセットを用意し、反復ステップに応じて
        どのエキスパートを使うか切り替えられるようにした。num_experts=1なら
        従来通り(share_weights/独立層のいずれか)、num_experts>1ならこちらが
        優先される。

        moe_routing="fixed"（既定・推奨）:
          ステップ番号でエキスパートを決定的に割り当てる（ブロック分割）。
          学習でルーティングを決めないため、データが少ない・状態空間が単純な
          設定にありがちな「ルーターが特定のエキスパートだけに偏って崩壊する」
          リスクが原理的に無い。それでいて「初期の反復」と「後期の反復」で
          別々の重みを使えるため、層ごとの役割分担の余地を残せる。
        moe_routing="learned":
          CLSトークンの現在表現からゲーティングネットワークが混合比率を
          学習する、より一般的なMoE。num_experts=4程度・状態空間も単純な
          ため、top-kのスパースルーティングではなく、全エキスパートを
          計算して混合比率で加重平均する密なMoE（num_experts=4程度なら
          計算コストの増加は無視できる範囲）にしている。崩壊しやすい点には注意。
        """
        super().__init__()
        self.share_weights = share_weights
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.moe_routing = moe_routing

        if use_step_embedding is None:
            use_step_embedding = share_weights or (num_experts > 1)
        self.use_step_embedding = use_step_embedding

        if num_experts > 1:
            # 【MoE的ステップルーティング】num_experts個の重みセットを用意
            self.experts = nn.ModuleList([
                RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
                for _ in range(num_experts)
            ])
            self.layers = None
            self.shared_layer = None

            if moe_routing == "fixed":
                # 各反復ステップ -> エキスパートindex をブロック分割で決定的に割当
                # 例: num_layers=8, num_experts=4 -> [0,0,1,1,2,2,3,3]
                assign = [(i * num_experts) // num_layers for i in range(num_layers)]
                self.register_buffer(
                    "expert_assignment", torch.tensor(assign, dtype=torch.long), persistent=False
                )
                self.router = None
            elif moe_routing == "learned":
                self.expert_assignment = None
                self.router = nn.Linear(d_model, num_experts)
            else:
                raise ValueError(f"unknown moe_routing: {moe_routing}")
        else:
            self.experts = None
            self.router = None
            self.expert_assignment = None
            if share_weights:
                self.shared_layer = RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
                self.layers = None
            else:
                self.layers = nn.ModuleList([
                    RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
                    for _ in range(num_layers)
                ])
                self.shared_layer = None

        if self.use_step_embedding:
            # 反復回数ぶんの「今が何回目か」を表すベクトルを学習する
            self.step_embedding = nn.Parameter(torch.randn(num_layers, d_model) * 0.02)
        else:
            self.step_embedding = None

    def _apply_moe_step(self, x, i, rope_cos, rope_sin, attn_mask):
        if self.moe_routing == "fixed":
            expert = self.experts[self.expert_assignment[i].item()]
            return expert(x, rope_cos, rope_sin, attn_mask=attn_mask)

        # moe_routing == "learned": CLSトークンの表現からゲーティング重みを決め、
        # 全エキスパートの出力を混合する（密なMoE）
        gate_logits = self.router(x[:, 0])                     # (B, num_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)      # (B, num_experts)
        expert_outs = torch.stack(
            [e(x, rope_cos, rope_sin, attn_mask=attn_mask) for e in self.experts],
            dim=0,
        )  # (num_experts, B, N, d_model)
        gate_weights = gate_weights.permute(1, 0).reshape(self.num_experts, -1, 1, 1)  # (E,B,1,1)
        return (expert_outs * gate_weights).sum(dim=0)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        if self.num_experts > 1:
            for i in range(self.num_layers):
                if self.use_step_embedding:
                    x = x + self.step_embedding[i].view(1, 1, -1)
                x = self._apply_moe_step(x, i, rope_cos, rope_sin, attn_mask)
        elif self.share_weights:
            for i in range(self.num_layers):
                if self.use_step_embedding:
                    x = x + self.step_embedding[i].view(1, 1, -1)
                x = self.shared_layer(x, rope_cos, rope_sin, attn_mask=attn_mask)
        else:
            for i, layer in enumerate(self.layers):
                if self.use_step_embedding:
                    x = x + self.step_embedding[i].view(1, 1, -1)
                x = layer(x, rope_cos, rope_sin, attn_mask=attn_mask)
        return x


# ================================================================
# 2. Transformer Actor-Critic Network (形状変換の完全防御)
# ================================================================
class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        in_channels=5,   # ch4=ゴールまでの正規化BFS距離（MazeEnv側の変更と対応）
        grid_size=5,
        d_model=64,
        nhead=4,
        # 変更: 既定値を「壁マスク使用＋多層＋重み共有」に変更。
        # 理由: measure_bfs_alignment の実測で、フルAttention+層2では
        # BFS最短方向との一致率が学習後もチャンスレベル(約50%)からほぼ
        # 改善しないことが確認された。フルAttentionは「1発で全マスを見れて
        # しまう」ため、隣接マスの価値を1ホップずつ伝播させるという計算を
        # 学習する動機が弱い。壁マスクで隣接マスのみに制限し、層数を
        # 迷路の直径程度(5x5なら8)まで増やすことで、「1層=1ホップの価値伝播」
        # を強制する。num_layersを8にするならday_modelは変えず重み共有で
        # パラメータ数の増加を抑える。
        num_layers=8,
        action_dim=4,
        hidden_size=128,
        wall_channel_idx=0,      # obs[0] = 壁マップ (1=壁, 0=通行可)
        distance_channel_idx=4,  # obs[4] = ゴールまでの正規化BFS距離（補助ロス用）
        use_wall_mask=True,      # 変更: 既定でON（理由は上記コメント参照）
        share_weights=True,      # 追加: Universal Transformer的に全層で重みを共有する
        use_step_embedding=None,  # 追加: Noneならshare_weightsと同じ値（重み共有時のみ既定でON）
        num_experts=1,            # 追加: 2以上でMoE的ステップルーティングを使う（share_weightsより優先）
        moe_routing="fixed",      # 追加: "fixed"=ステップ番号で決定的に割当（推奨） / "learned"=学習的ゲーティング
        dropout=0.1,              # 追加: アブレーション実験用に外側から調整できるように
        rope_base=10000.0,       # RoPEの周波数の基数
    ):
        super().__init__()

        head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model は nhead で割り切れる必要があります"
        assert head_dim % 4 == 0, (
            "2D RoPEを使うには head_dim(=d_model//nhead) が4の倍数である必要があります。"
            f"現在 d_model={d_model}, nhead={nhead} -> head_dim={head_dim}"
        )

        self.grid_size = grid_size
        self.in_channels = in_channels
        self.num_tokens = grid_size * grid_size
        self.nhead = nhead
        self.wall_channel_idx = wall_channel_idx
        self.distance_channel_idx = distance_channel_idx
        self.d_model = d_model
        self.use_wall_mask = use_wall_mask

        self.embedding = nn.Linear(in_channels, d_model)

        # 変更: 学習可能な絶対位置埋め込み(row_embedding/col_embedding/
        # cls_pos_embedding)は廃止。位置情報は2D RoPEがAttention内で
        # 直接扱うため、トークン特徴量に別途加算する必要がなくなった。
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 事前計算: 各セルindex -> (row_idx, col_idx) の対応をbufferとして保持
        row_idx, col_idx = self._build_grid_indices(grid_size)
        self.register_buffer("row_idx", row_idx, persistent=False)  # (N,)
        self.register_buffer("col_idx", col_idx, persistent=False)  # (N,)

        # 追加: 2D RoPEのcos/sinテーブル（CLS分1つ+グリッドN個 = N+1トークン分）
        # グリッド構造は固定なので、モデル構築時に一度だけ計算しbufferとして保持する。
        rope_cos, rope_sin = build_2d_rope_tables(grid_size, head_dim, base=rope_base)
        self.register_buffer("rope_cos", rope_cos, persistent=False)  # (N+1, head_dim)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # 変更: nn.TransformerEncoder(nn.TransformerEncoderLayer) を廃止し、
        # RoPEをQ/Kに適用できる自前実装のEncoderに置き換える。
        self.transformer = RoPETransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            num_layers=num_layers,
            dropout=dropout,
            share_weights=share_weights,
            use_step_embedding=use_step_embedding,
            num_experts=num_experts,
            moe_routing=moe_routing,
        )

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        # 追加: 各マスのゴールまでの距離を予測する補助ヘッド（学習の補助信号用）。
        # obs[4]としてすでに正解の距離が入力されているが、Embedding+Attentionを
        # 経た後のトークン表現がその情報を忠実に保持できているかは別問題。
        # 「各マスのトークン表現から、そのマスの正しい距離を再現できるか」を
        # 直接教師あり学習させることで、間接的で疎なPPOの報酬信号だけに頼らず、
        # 距離情報を正確に読み取る/伝播させる表現をAttention側に強制する。
        self.distance_head = nn.Linear(d_model, 1)

        # 隣接ペア（上下左右）を事前計算し、モデルのbufferとして登録
        idx_i, idx_j = self._build_neighbor_indices(grid_size)
        self.register_buffer("neighbor_i", idx_i, persistent=False)
        self.register_buffer("neighbor_j", idx_j, persistent=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _build_grid_indices(grid_size):
        """トークン順(row-major: i = r*W + c)に対応する row_idx, col_idx を返す"""
        H = W = grid_size
        rows, cols = [], []
        for r in range(H):
            for c in range(W):
                rows.append(r)
                cols.append(c)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)

    @staticmethod
    def _build_neighbor_indices(grid_size):
        H = W = grid_size
        pairs_i, pairs_j = [], []
        for r in range(H):
            for c in range(W):
                i = r * W + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        pairs_i.append(i)
                        pairs_j.append(nr * W + nc)
        return torch.tensor(pairs_i, dtype=torch.long), torch.tensor(pairs_j, dtype=torch.long)

    def _build_attn_mask(self, wall_map, agent_pos_flat, B, device):
        """
        wall_map: (B, H, W)
        agent_pos_flat: (B,) 各バッチの現在位置のフラットインデックス

        【重要な制約・use_wall_mask=Falseにした理由】
        このmaskはCLSトークンが「現在位置から1ホップ隣接するマスの情報」しか
        直接受け取れない設計になっている（adjacencyは上下左右1マスのみ）。
        num_layers=2の場合、CLSが集約できる情報は実質2ホップ分に限られ、
        5x5迷路で必要になりうる距離（最大8マス程度）の経路情報を
        表現できない。これが「数マス先の迂回が必要な壁を避けられない」
        原因になっていたため、既定では使わない。

        代わりにMazeEnv側でBFS距離マップをobs[4]として直接与えることで、
        受容野に依存せず各セルがゴール方向の情報を持てるようにしている。

        use_wall_mask=True にすると従来通りこのmaskを使う（アブレーション用）。
        使う場合は num_layers を迷路の直径をカバーできる程度まで
        増やすことを推奨（5x5なら num_layers=4〜6程度）。

        戻り値は scaled_dot_product_attention 向けの bool mask
        （True=attend可, shape (B,1,N+1,N+1)。全headにブロードキャストされる）。
        nn.MultiheadAttentionの旧APIとは True/False の意味が逆なので注意。
        """
        H = W = self.grid_size
        N = self.num_tokens
        wall_flat = wall_map.reshape(B, N)

        adjacency = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        not_wall_i = (wall_flat[:, self.neighbor_i] == 0)
        not_wall_j = (wall_flat[:, self.neighbor_j] == 0)
        connected = not_wall_i & not_wall_j
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.neighbor_i.shape[0])
        adjacency[batch_idx, self.neighbor_i.expand(B, -1), self.neighbor_j.expand(B, -1)] = connected
        diag_idx = torch.arange(N, device=device)
        adjacency[:, diag_idx, diag_idx] = True

        full_adj = torch.zeros(B, N + 1, N + 1, dtype=torch.bool, device=device)
        full_adj[:, 1:, 1:] = adjacency

        # CLS(index 0) は「現在位置のマスと同じ接続性」を持たせる
        # -> 現在位置から到達可能なマスの情報だけを、CLSは直接受け取れる
        batch_range = torch.arange(B, device=device)
        cls_connectivity = adjacency[batch_range, agent_pos_flat]  # (B, N) 現在位置の行を流用
        full_adj[:, 0, 1:] = cls_connectivity
        full_adj[:, 1:, 0] = cls_connectivity
        full_adj[:, 0, 0] = True

        # scaled_dot_product_attention は True=attend可 の意味なので full_adj をそのまま返す
        return full_adj.unsqueeze(1)  # (B, 1, N+1, N+1) -> head方向にブロードキャスト

    def forward(self, x, return_aux=False):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        assert x.dim() == 4, f"Expected 4D (B,C,H,W) input, got {tuple(x.shape)}"
        B, C, H, W = x.shape

        wall_map = x[:, self.wall_channel_idx, :, :]  # (B, H, W)  ※permute前に取得
        agent_map = x[:, 3, :, :]  # (B, H, W) エージェント位置チャンネル
        # 補助ロス用に、埋め込み前の生の距離チャンネルを保持しておく
        distance_target = x[:, self.distance_channel_idx, :, :].reshape(B, -1)  # (B, N)

        agent_pos_flat = agent_map.reshape(B, -1).argmax(dim=1)  # (B,)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)  # (B, 25, C)

        x = self.embedding(x)  # (B, 25, d_model)

        # 変更: 絶対位置埋め込みの加算を廃止（2D RoPEに置き換えたため不要）。
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 26, d_model)

        if self.use_wall_mask:
            attn_mask = self._build_attn_mask(wall_map, agent_pos_flat, B, x.device)
        else:
            attn_mask = None

        x = self.transformer(x, self.rope_cos, self.rope_sin, attn_mask=attn_mask)

        cls_out = x[:, 0]
        grid_out = x[:, 1:]  # (B, N, d_model)  CLSを除いた各マスのトークン表現

        logits = self.actor_head(cls_out)
        value = self.critic_head(cls_out)

        if return_aux:
            # 追加: 各マスのトークン表現から、そのマスのゴールまでの距離を予測する
            distance_pred = self.distance_head(grid_out).squeeze(-1)  # (B, N)
            return logits, value.squeeze(-1), distance_pred, distance_target

        return logits, value.squeeze(-1)


# ================================================================
# 2.5. Value Iteration Network (VIN) Actor-Critic
# ================================================================
# 背景: measure_bfs_alignment の実測で、Transformer版は学習後も
# BFS最短方向との一致率がチャンスレベル(約50%)からほぼ改善しないことが
# 確認された。これは「距離の情報(ch4)は入力されているのに、隣接マスの
# 値を精密に比較して最善方向を選ぶ」という計算自体を、Attention
# （本質的にはsoftmaxによる"ぼんやりした加重平均"）ではうまく学習できて
# いないことを示唆している。
#
# VINはこの「値を伝播させて比較する」計算そのものを、畳み込み+maxという
# 明確な演算としてネットワーク構造に組み込む(Tamar et al., 2016)。
#   1. 報酬マップR: 壁チャンネル・ゴールチャンネルから直接構成する
#      （すでに正確な情報があるため学習させず、学習容量を「伝播」に集中させる）
#   2. 価値の伝播: Q = conv([R; V]), V = max_a(Q) を、重み共有した同じ
#      畳み込み層でK回繰り返す（Bellman backupのK回反復に相当）
#   3. 現在のエージェント位置でのQ値ベクトルを取り出し、それを方策・価値の
#      出力に変換する
class VINActorCritic(nn.Module):
    def __init__(
        self,
        in_channels=5,          # 参考情報として保持（forwardでは主要3チャンネルのみ使用）
        grid_size=5,
        action_dim=4,
        vi_iterations=15,       # 5x5迷路の対角線分(約8)より十分大きい反復回数
        hidden_size=64,
        wall_channel_idx=0,
        goal_channel_idx=2,
        agent_channel_idx=3,
        wall_reward=-10.0,      # 壁マスの報酬（強い負：伝播が壁を避けるよう学習させる目印）
        goal_reward=10.0,       # ゴールマスの報酬
        step_reward=-0.1,       # それ以外のマスの報酬（歩くコスト）
    ):
        super().__init__()
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.vi_iterations = vi_iterations
        self.wall_channel_idx = wall_channel_idx
        self.goal_channel_idx = goal_channel_idx
        self.agent_channel_idx = agent_channel_idx
        self.wall_reward = wall_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        # 【VINの核】価値伝播を担う畳み込み層。
        # 入力2ch(R, V) -> 出力action_dim ch（各行動を選んだ場合のQ値マップ）。
        # このカーネルをvi_iterations回、重み共有したまま繰り返し適用することで、
        # 「隣のマスの価値を見て、自分の価値を更新する」というBellman backupを
        # ネットワーク構造として直接表現する。
        self.q_conv = nn.Conv2d(2, action_dim, kernel_size=3, padding=1, bias=False)

        # 現在位置でのQ値ベクトル(action_dim次元)から方策・価値を出す最終ヘッド
        self.actor_head = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_reward_map(self, x):
        """壁チャンネル・ゴールチャンネルから報酬マップRを直接構成する（学習不要）"""
        B, C, H, W = x.shape
        wall = x[:, self.wall_channel_idx]   # (B,H,W)
        goal = x[:, self.goal_channel_idx]   # (B,H,W)

        R = torch.full((B, H, W), self.step_reward, device=x.device, dtype=x.dtype)
        R = torch.where(wall > 0.5, torch.full_like(R, self.wall_reward), R)
        R = torch.where(goal > 0.5, torch.full_like(R, self.goal_reward), R)
        return R.unsqueeze(1)  # (B,1,H,W)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4, f"Expected 4D (B,C,H,W) input, got {tuple(x.shape)}"
        B, C, H, W = x.shape

        R = self.build_reward_map(x)                       # (B,1,H,W)
        V = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # 【価値反復】重み共有した同じ畳み込みをK回繰り返す
        for _ in range(self.vi_iterations):
            q = self.q_conv(torch.cat([R, V], dim=1))       # (B, action_dim, H, W)
            V, _ = q.max(dim=1, keepdim=True)                # (B,1,H,W)

        # 収束後の価値マップで最後にもう一度Qマップを計算
        q_final = self.q_conv(torch.cat([R, V], dim=1))     # (B, action_dim, H, W)

        agent_map = x[:, self.agent_channel_idx]             # (B,H,W)
        agent_flat = agent_map.reshape(B, -1).argmax(dim=1)  # (B,)
        agent_r = agent_flat // W
        agent_c = agent_flat % W

        batch_idx = torch.arange(B, device=x.device)
        psi = q_final[batch_idx, :, agent_r, agent_c]        # (B, action_dim) 現在位置でのQ値

        logits = self.actor_head(psi)
        value = self.critic_head(psi).squeeze(-1)
        return logits, value


# ================================================================
# 3. PPO Agent
# ================================================================
def _init_weights(m):
    """直交初期化（Orthogonal Initialization）の基本関数"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TransformerPPOAgent:

    def __init__(
        self,
        env,
        in_channels=5,   # 変更: MazeEnvの距離チャンネル追加に合わせる
        grid_size=5,
        d_model=64,
        nhead=4,
        num_layers=8,          # 変更: TransformerActorCriticの新しい既定値に合わせる
        action_dim=4,
        hidden_size=128,
        use_wall_mask=True,    # 変更: 既定でON（理由はTransformerActorCritic側のコメント参照）
        share_weights=True,    # 追加: Universal Transformer的に全層で重みを共有する
        use_step_embedding=None,  # 追加: Noneならshare_weightsと同じ値
        num_experts=1,          # 追加: 2以上でMoE的ステップルーティングを使う
        moe_routing="fixed",    # 追加: "fixed"（推奨） / "learned"
        dropout=0.1,               # 追加: アブレーション実験用
        distance_loss_coef=0.1,  # 追加: 距離予測補助ロスの重み
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=10,
        batch_size=32,
        entropy_coef=0.03,
        entropy_coef_final=None,  # 追加: 指定するとentropy_coefから線形に減衰させる（探索→活用）
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        path_save="transformer_ppo.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        # 追加: アニーリング用に開始値・終了値を保持。entropy_coef_final未指定なら
        # 従来通り定数（アニーリングなし）。
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_final = (
            entropy_coef_final if entropy_coef_final is not None else entropy_coef
        )
        self.value_loss_coef = value_loss_coef
        self.distance_loss_coef = distance_loss_coef
        self.max_grad_norm = max_grad_norm
        self.path_save = path_save
        self.device = device

        # ネットワークの初期化
        self.policy = TransformerActorCritic(
            in_channels=in_channels,
            grid_size=grid_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_dim=action_dim,
            hidden_size=hidden_size,
            use_wall_mask=use_wall_mask,
            share_weights=share_weights,
            use_step_embedding=use_step_embedding,
            num_experts=num_experts,
            moe_routing=moe_routing,
            dropout=dropout,
        )
        self.policy.to(device)

        # -------------------------------------------------------------
        # 【コツ1】直交初期化と出力層ゲインの調整
        # -------------------------------------------------------------
        self.policy.apply(_init_weights)
        self._apply_head_gain_initialization()

        # 修正: 以前はここに来る前に無条件で self.load_model(path_save) を呼んでおり、
        # チェックポイントファイルが存在しない初回実行時に torch.load が
        # FileNotFoundError で落ちる可能性があった。os.path.exists 判定後の
        # 1箇所にまとめる。
        if os.path.exists(self.path_save):
            self.load_model(self.path_save)
            print("Successfully loaded model.")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = PPORolloutBuffer()

    def _apply_head_gain_initialization(self):
        """Actorヘッド（0.01）とCriticヘッド（1.0）のゲインを個別に調整"""
        # ネットワーク構造に合わせてヘッド属性名を取得（一般的な命名に対応）
        actor_head = getattr(
            self.policy, "actor", getattr(self.policy, "action_head", None)
        )
        critic_head = getattr(
            self.policy, "critic", getattr(self.policy, "value_head", None)
        )

        if actor_head is not None:
            last_layer = (
                actor_head[-1]
                if isinstance(actor_head, nn.Sequential)
                else actor_head
            )
            if hasattr(last_layer, "weight"):
                nn.init.orthogonal_(last_layer.weight, gain=0.01)

        if critic_head is not None:
            last_layer = (
                critic_head[-1]
                if isinstance(critic_head, nn.Sequential)
                else critic_head
            )
            if hasattr(last_layer, "weight"):
                nn.init.orthogonal_(last_layer.weight, gain=1.0)

    def _preprocess_state(self, state_img):
        if isinstance(state_img, np.ndarray):
            state_img = torch.from_numpy(state_img).float()
        # バッチ次元（Dim 0）がない場合は追加
        if state_img.dim() == 3:
            state_img = state_img.unsqueeze(0)
        return state_img.to(self.device)

    def select_action(self, state_img, deterministic=False):
        """
        deterministic=False（既定）: 学習時の探索用。Categorical分布からサンプリングする。
        deterministic=True: 評価用。マスク後のロジットが最大の行動を決定的に選ぶ
        （残存エントロピーによる「たまたま外す」を排除し、方策が本当に学習できているかを
        正しく測るために使う）。
        """
        self.policy.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_state(state_img)
            logits, value = self.policy(state_tensor)

            # ナンバー・オーバーフロー保護
            logits = torch.clamp(logits, min=-20.0, max=20.0)

            # 壁・盤外に向かう行動をマスクしてから分布を作る
            valid_mask = compute_valid_action_mask(state_tensor)
            logits = apply_action_mask(logits, valid_mask)

            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate(self, num_episodes=200, max_steps_per_episode=60,
                 maze_change=True, min_distance=None, max_distance=None,
                 deterministic=True):
        """
        学習を行わず、方策の実力を測るための評価専用メソッド。
        deterministic=Trueならgreedy行動選択（サンプリングの運を排除）で成功率を測る。
        deterministic=Falseなら学習時と同じ確率的サンプリングでの成功率を測れるので、
        両方を比べることで「サンプリングのブレによる失敗」がどれだけあるかが分かる。
        """
        successes = 0
        for _ in range(num_episodes):
            self.env.reset(
                maze_change=maze_change,
                min_distance=min_distance, max_distance=max_distance,
            )
            state_img = self.env.get_image_observation()
            done = False
            for _ in range(max_steps_per_episode):
                action, _, _ = self.select_action(state_img, deterministic=deterministic)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    _, _, done = step_result[:3]
                state_img = self.env.get_image_observation()
                if done:
                    successes += 1
                    break
        return successes / num_episodes

    def measure_bfs_alignment(self, num_episodes=100, max_steps_per_episode=60,
                               maze_change=True, min_distance=None, max_distance=None,
                               deterministic=True):
        """
        診断用: 各ステップで「モデルが選んだ行動」が、BFS距離マップが示す
        真の最短方向（有効な隣接マスの中で最もゴールに近いマスへの行動）と
        どれだけ一致しているかを測る。

        これにより「情報(ch4)は入力されているのに、モデルがそれを正確に
        読み取れていないのでは」という仮説を実データで検証できる。
        一致率が低いなら、情報を"読む"こと自体がボトルネック
        （Attentionの精密な比較が苦手、報酬信号が間接的すぎる、等）である
        可能性が高く、VINのようなアーキテクチャ変更が効きやすいと判断できる。
        一致率が高いのに成功率が低いなら、別の要因（実行時のブレ等）を疑うべき。
        """
        total_steps = 0
        aligned_steps = 0

        # env.step の action定義: 0=上 1=下 2=左 3=右
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for _ in range(num_episodes):
            self.env.reset(
                maze_change=maze_change,
                min_distance=min_distance, max_distance=max_distance,
            )
            state_img = self.env.get_image_observation()
            done = False

            for _ in range(max_steps_per_episode):
                x, y = self.env.state
                cur_dist = self.env.dist_map[x][y]

                # BFS距離マップ上で、有効な移動先の中から最も距離が小さい
                # （＝ゴールに最も近い）行動集合を「正解」とする
                best_actions = []
                best_dist = None
                for a, (dr, dc) in deltas.items():
                    nx, ny = x + dr, y + dc
                    if not self.env._is_valid_move((nx, ny)):
                        continue
                    d = self.env.dist_map[nx][ny]
                    if d is None:
                        continue
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_actions = [a]
                    elif d == best_dist:
                        best_actions.append(a)

                action, _, _ = self.select_action(state_img, deterministic=deterministic)

                if best_actions:  # 有効な移動先が無いマスは判定対象から除外
                    total_steps += 1
                    if action in best_actions:
                        aligned_steps += 1

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    _, _, done = step_result[:3]
                state_img = self.env.get_image_observation()
                if done:
                    break

        alignment_rate = aligned_steps / total_steps if total_steps > 0 else None
        return {
            "total_steps": total_steps,
            "aligned_steps": aligned_steps,
            "alignment_rate": alignment_rate,
        }

    def _goal_open_count(self):
        """
        現在のself.envの迷路について、ゴールマスの「開口数」
        （壁でも盤外でもない隣接マスの数、0〜4）を数える。
        1以下なら「ゴールがほぼ壁で囲まれている（進入路が実質1つ）」とみなせる。
        """
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        gx, gy = self.env.goal
        return sum(
            1 for dr, dc in deltas
            if self.env._is_valid_move((gx + dr, gy + dc))
        )

    def measure_bfs_alignment_by_difficulty(
        self,
        num_episodes_per_category=100,
        max_steps_per_episode=60,
        near_goal_threshold=3,
        deterministic=True,
        hard_open_count_max=1,
        max_attempts_multiplier=50,
    ):
        """
        診断用: measure_bfs_alignment を「ゴールの開口数」で難易度別に分けて
        測定する。

        - "easy": ゴールの隣接マスのうち、壁でないマスがhard_open_count_maxより
          多い（進入路が複数ある、比較的開けたゴール）
        - "hard": ゴールの隣接マスのうち、壁でないマスがhard_open_count_max以下
          （進入路が実質1つしかない、壁でほぼ囲まれたゴール）

        さらに、各カテゴリについて「ゴールからnear_goal_threshold手以内」に
        限定した一致率も別途集計する。これは「全体としては改善したが、
        ゴールが壁でほぼ囲まれるレアケースの、しかもまさにゴール目前の局面
        だけ苦手なまま」という仮説をピンポイントで検証するため。

        戻り値: {"easy": {...}, "hard": {...}} で、それぞれ
          episodes, overall_alignment_rate, near_goal_alignment_rate,
          total_steps, near_goal_steps を含む。

        注意: "hard"に該当する迷路（壁密度0.3の既定設定では）は出現頻度が
        低いことがあるため、規定数集まるまで自動でエピソードを追加生成する。
        max_attempts_multiplier に達すると打ち切って警告を出す。
        """
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        stats = {
            "easy": {"total": 0, "aligned": 0, "near_goal_total": 0,
                     "near_goal_aligned": 0, "episodes": 0},
            "hard": {"total": 0, "aligned": 0, "near_goal_total": 0,
                     "near_goal_aligned": 0, "episodes": 0},
        }

        max_attempts = num_episodes_per_category * max_attempts_multiplier
        attempts = 0

        while (stats["easy"]["episodes"] < num_episodes_per_category
               or stats["hard"]["episodes"] < num_episodes_per_category):
            attempts += 1
            if attempts > max_attempts:
                print(
                    "警告: 試行上限に達しました。"
                    f"easy={stats['easy']['episodes']}件, hard={stats['hard']['episodes']}件 "
                    f"（目標各{num_episodes_per_category}件）で打ち切ります。"
                    "hardカテゴリが集まりにくい場合、迷路生成のwall_densityを"
                    "上げるか、num_episodes_per_categoryを減らしてください。"
                )
                break

            self.env.reset(maze_change=True)
            open_count = self._goal_open_count()
            category = "hard" if open_count <= hard_open_count_max else "easy"

            if stats[category]["episodes"] >= num_episodes_per_category:
                continue  # このカテゴリは既に規定数集まっている

            state_img = self.env.get_image_observation()
            done = False

            for _ in range(max_steps_per_episode):
                x, y = self.env.state
                cur_dist = self.env.dist_map[x][y]

                best_actions = []
                best_dist = None
                for a, (dr, dc) in deltas.items():
                    nx, ny = x + dr, y + dc
                    if not self.env._is_valid_move((nx, ny)):
                        continue
                    d = self.env.dist_map[nx][ny]
                    if d is None:
                        continue
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_actions = [a]
                    elif d == best_dist:
                        best_actions.append(a)

                action, _, _ = self.select_action(state_img, deterministic=deterministic)

                if best_actions:
                    stats[category]["total"] += 1
                    is_aligned = action in best_actions
                    if is_aligned:
                        stats[category]["aligned"] += 1

                    if cur_dist is not None and cur_dist <= near_goal_threshold:
                        stats[category]["near_goal_total"] += 1
                        if is_aligned:
                            stats[category]["near_goal_aligned"] += 1

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    _, _, done = step_result[:3]
                state_img = self.env.get_image_observation()
                if done:
                    break

            stats[category]["episodes"] += 1

        result = {}
        for category, s in stats.items():
            result[category] = {
                "episodes": s["episodes"],
                "overall_alignment_rate": (
                    s["aligned"] / s["total"] if s["total"] > 0 else None
                ),
                "near_goal_alignment_rate": (
                    s["near_goal_aligned"] / s["near_goal_total"]
                    if s["near_goal_total"] > 0 else None
                ),
                "total_steps": s["total"],
                "near_goal_steps": s["near_goal_total"],
            }
        return result

    def update(self, next_state_img, done):
        self.policy.train()

        # Bootstrap用の次状態価値の取得
        with torch.no_grad():
            next_state_tensor = self._preprocess_state(next_state_img)
            _, next_value = self.policy(next_state_tensor)
            next_value = next_value.squeeze().item() if not done else 0.0

        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]

        # -------------------------------------------------------------
        # GAE (Generalized Advantage Estimation) の過去遡及計算
        # -------------------------------------------------------------
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * non_terminal
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        # テンソルへの変換
        states_tensor = torch.cat(
            [self._preprocess_state(s) for s in self.buffer.states], dim=0
        )
        actions_tensor = torch.tensor(
            self.buffer.actions, dtype=torch.long
        ).to(self.device)
        old_log_probs_tensor = torch.tensor(
            self.buffer.log_probs, dtype=torch.float
        ).to(self.device)
        old_values_tensor = torch.tensor(
            self.buffer.values, dtype=torch.float
        ).to(self.device)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float
        ).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float).to(
            self.device
        )

        # -------------------------------------------------------------
        # 【コツ2-2】アドバンテージ標準化（Advantage Normalization）
        # -------------------------------------------------------------
        if len(advantages_tensor) > 1:
            adv_std = advantages_tensor.std()
            if not torch.isnan(adv_std) and adv_std > 1e-8:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                ) / (adv_std + 1e-8)
            else:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                )
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        dataset_size = len(self.buffer.states)

        # PPOエポックループ
        for _ in range(self.ppo_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = states_tensor[batch_idx]
                b_actions = actions_tensor[batch_idx]
                b_old_log_probs = old_log_probs_tensor[batch_idx]
                b_old_values = old_values_tensor[batch_idx]
                b_advantages = advantages_tensor[batch_idx]
                b_returns = returns_tensor[batch_idx]

                logits, new_values, distance_pred, distance_target = self.policy(
                    b_states, return_aux=True
                )
                new_values = new_values.squeeze(-1)

                # Logits の数値的安定化
                logits = torch.clamp(logits, min=-20.0, max=20.0)

                # 追加: ロールアウト時（select_action）と同じマスクをここでも適用しないと、
                # 保存済みold_log_probs（マスクあり）と学習時のlog_probs（マスクなし）が
                # 食い違い、PPOのratio計算が歪む。
                valid_mask = compute_valid_action_mask(b_states)
                logits = apply_action_mask(logits, valid_mask)

                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Ratio と Clipped Surrogate Loss の計算
                log_ratio = new_log_probs - b_old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratios = torch.exp(log_ratio)

                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(
                        ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # -------------------------------------------------------------
                # 【コツ1-1】Value Function（Critic）のクリッピング Loss
                # -------------------------------------------------------------
                v_clipped = b_old_values + torch.clamp(
                    new_values - b_old_values, -self.clip_eps, self.clip_eps
                )
                v_loss_unclipped = (new_values - b_returns) ** 2
                v_loss_clipped = (v_clipped - b_returns) ** 2
                value_loss = 0.5 * torch.max(
                    v_loss_unclipped, v_loss_clipped
                ).mean()

                # -------------------------------------------------------------
                # 追加: 距離予測補助ロス
                # 各マスのトークン表現から、そのマスの正しいゴールまでの距離
                # (obs[4]の値、壁マスを除く)を予測できるかをMSEで直接教師あり学習する。
                # PPOの疎で間接的な報酬信号だけに頼らず、「距離情報を正確に読み取り、
                # 伝播させる」表現をAttention側に強制する狙い。
                # -------------------------------------------------------------
                wall_flat = b_states[:, 0].reshape(b_states.shape[0], -1)  # (B,N)
                non_wall_mask = (wall_flat < 0.5).float()
                sq_err = (distance_pred - distance_target) ** 2
                distance_loss = (sq_err * non_wall_mask).sum() / non_wall_mask.sum().clamp(min=1.0)

                # トータル Loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.distance_loss_coef * distance_loss
                )

                # 数値異常のガード
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        "Warning: Loss is NaN/Inf, skipping backward pass."
                    )
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # -------------------------------------------------------------
                # 【コツ3-2】勾配クリッピング
                # -------------------------------------------------------------
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.clear()

    def train(
        self,
        num_episodes=1000,
        max_steps_per_episode=100,
        update_horizon=512,
        maze_change=True,
        path=None,
        log_interval=10,
        use_curriculum=False,
        curriculum_start_distance=3,   # 追加: 最初はこの距離までの迷路だけ出す
        curriculum_success_threshold=0.7,  # 追加: 直近window内の成功率がこれを超えたら難易度UP
        curriculum_window=20,
        use_prioritized_replay=False,   # 追加: 苦手な迷路を優先出題するプールを使うか
        pool_size=100,                  # プールに保持する迷路の最大数
        pool_history_len=10,            # 1迷路あたり直近何回分の成功/失敗を見るか
        pool_new_maze_prob=0.2,         # プールが満杯後、新規ランダム迷路を注入する確率
    ):
        save_path = path if path is not None else self.path_save
        episode_rewards = []
        success_flags = []  # 直近のゴール到達可否を記録（カリキュラム判定用。難易度UP時にリセットされる）
        success_history = []  # 追加: カリキュラムでリセットされない、全エピソード分の完全な記録（分散分析用）

        # 追加: カリキュラム学習用の現在の難易度上限（スタート-ゴール間距離）
        current_max_distance = curriculum_start_distance if use_curriculum else None

        # 追加: 苦手迷路プール。curriculumと併用した場合、新規迷路の注入は
        # その時点のcurriculumの難易度上限に従う（両方の恩恵を受けられる）。
        pool = MazePool(
            pool_size=pool_size,
            history_len=pool_history_len,
            new_maze_prob=pool_new_maze_prob,
        ) if use_prioritized_replay else None
        self.maze_pool = pool  # 学習後にも参照できるようエージェントに保持

        for episode in range(num_episodes):
            # 追加: エントロピー係数を線形にアニーリング（探索→活用へ）
            progress = episode / max(1, num_episodes - 1)
            self.entropy_coef = (
                self.entropy_coef_start
                + (self.entropy_coef_final - self.entropy_coef_start) * progress
            )

            current_entry = None
            if use_prioritized_replay:
                if pool.should_inject_new():
                    # 新規ランダム迷路を生成してプールに追加（curriculum併用時は
                    # その時点の難易度上限に従う）
                    self.env.generate_random_maze(
                        rows=self.env.rows, cols=self.env.cols,
                        max_distance=current_max_distance if use_curriculum else None,
                    )
                    current_entry = pool.add(self.env.maze, self.env.start, self.env.goal)
                else:
                    # プールから苦手度に応じた確率で1つ選ぶ
                    current_entry = pool.sample()
                maze_override = (
                    current_entry["maze"], current_entry["start"], current_entry["goal"]
                )
                state_pos = self.env.reset(maze_change=False, maze_override=maze_override)
            elif use_curriculum:
                state_pos = self.env.reset(
                    maze_change=maze_change, max_distance=current_max_distance
                )
            else:
                state_pos = self.env.reset(maze_change=maze_change)
            state_img = self.env.get_image_observation()
            episode_reward = 0
            episode_done = False
            next_state_img = state_img  # timeoutでstepが1回も回らない異常系向けの保険

            for step in range(max_steps_per_episode):
                action, log_prob, value = self.select_action(state_img)

                # step の戻り値形式（3要素/5要素）の柔軟な受け取り
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state_pos, reward, terminated, truncated, _ = (
                        step_result
                    )
                    done = terminated or truncated
                else:
                    next_state_pos, reward, done = step_result[:3]

                next_state_img = self.env.get_image_observation()

                self.buffer.push(
                    state_img, action, log_prob, reward, done, value
                )

                state_pos = next_state_pos
                state_img = next_state_img
                episode_reward += reward
                episode_done = done

                # horizon到達 or done(=ゴール到達)の場合はここで更新。
                if len(self.buffer.states) >= update_horizon or done:
                    self.update(next_state_img, done=done)

                if done:
                    break

            # timeoutで終わったエピソードもここで必ずupdateし、bufferを毎エピソードで
            # クリアする（迷路をまたいだGAE汚染を防ぐ）。
            if len(self.buffer.states) > 0:
                self.update(next_state_img, done=episode_done)

            # 追加: このエピソードで使った迷路の成功/失敗をプールに記録
            if use_prioritized_replay and current_entry is not None:
                pool.record_result(current_entry, success=episode_done)

            episode_rewards.append(episode_reward)
            success_flags.append(1 if episode_done else 0)
            success_history.append(1 if episode_done else 0)

            if episode % log_interval == 0:
                window = success_flags[-log_interval:]
                success_rate = sum(window) / len(window)
                curriculum_info = (
                    f", 難易度(max_distance)={current_max_distance}"
                    if use_curriculum else ""
                )
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, "
                    f"直近{len(window)}エピソードのゴール到達率: {success_rate:.0%}, "
                    f"entropy_coef={self.entropy_coef:.4f}{curriculum_info}"
                )
                if use_prioritized_replay:
                    stats = pool.stats(top_k=5)
                    if stats["overall_success_rate"] is not None:
                        hardest_str = ", ".join(
                            f"id{h['id']}(成功率{h['recent_success_rate']:.0%}, 試行{h['attempts']}回)"
                            for h in stats["hardest"]
                        )
                        print(
                            f"  [プール] 試行済み{stats['num_tried']}/{stats['pool_size']}件, "
                            f"プール全体の成功率={stats['overall_success_rate']:.0%}"
                        )
                        print(f"  [プール] 苦手Top{len(stats['hardest'])}: {hardest_str}")

            # 追加: カリキュラムの難易度更新判定
            if use_curriculum and len(success_flags) >= curriculum_window:
                recent = success_flags[-curriculum_window:]
                recent_success_rate = sum(recent) / len(recent)
                if recent_success_rate >= curriculum_success_threshold:
                    current_max_distance += 1
                    print(
                        f"[curriculum] 直近{curriculum_window}エピソードの成功率"
                        f"{recent_success_rate:.0%} >= 閾値。"
                        f"難易度を引き上げ: max_distance={current_max_distance}"
                    )
                    success_flags = []  # 難易度が変わったら成功率をリセットして再計測

            if episode % 100 == 0 and episode > 0:
                print(f"Episode {episode}, save model to {save_path}")
                self.save_model(save_path)

        print("Training finished.")
        return episode_rewards, success_history

    def save_model(self, path):
        self.policy.cpu()
        torch.save(self.policy.state_dict(), path)
        self.policy.to(self.device)

    def load_model(self, path):
        # 修正: self.load_model_safely は存在しないメソッド名だったため
        # hasattr(self, "load_model_safely") は常にFalseとなり、
        # 「形状不一致レイヤーをスキップする安全ロード」が一度も
        # 使われていなかった。モジュール関数を直接呼ぶように修正。
        # これにより、今回のようにモデル構造（in_channels等）を変更した後でも、
        # 形状が一致するレイヤーだけを読み込んで継続学習できる。
        load_model_safely(self.policy, path, device=self.device)
