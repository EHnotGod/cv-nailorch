import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from nailorch.optimizers import *
from nailorch import *
import nailorch.functions as F
import nailorch.layers as L
import numpy as np
from tokenizers import Tokenizer
import os
import json

# %%
txt = ""
with open(r"../../data/st.txt", encoding="utf-8") as f:
    txt += f.read()
tokenizer = Tokenizer.from_file(os.path.join(r"./chinese_bbpe", "tokenizer.json"))
encoded = tokenizer.encode(txt)
ids = encoded.ids

print(ids[:10])


# %%
class GPT2SeqDataset(Dataset):
    def __init__(self, ids, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        self.data = []
        self.label = []
        # 生成连续 seq_len 的片段
        for i in range(0, len(ids) - seq_len):
            x = ids[i: i + seq_len]
            t = ids[i + 1: i + seq_len + 1]
            self.data.append(x)
            self.label.append(t)
        self.data = np.array(self.data, dtype=np.int32)
        self.label = np.array(self.label, dtype=np.int32)


class GPTDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, gpu=False, shuffle=True):
        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size
                       for i in range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]
        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])
        self.iteration += 1
        return x, t


# %%
class GPT2(Model):
    def __init__(self, n_vocab=300, n_seq=60, n_embd=60, hp_blocks=[(60, 2), (60, 2)]):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_seq = n_seq
        self.n_embd = n_embd

        self.wte = L.EmbedID(self.n_vocab, n_embd)
        self.wpe = Parameter(None, name='wpe')

        self.blocks = [TransformerBlock(*hp_block) for hp_block in hp_blocks]
        for i, blk in enumerate(self.blocks):
            setattr(self, f'block_{i}', blk)  # 注册为属性
        self.ln_f = L.LayerNorm()

        self.lm_head = L.Linear(in_size=self.n_embd, out_size=self.n_vocab)

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        if self.wpe.data is None:
            self.wpe.data = xp.random.randn(self.n_seq, self.n_embd).astype('float32')

    def __call__(self, x):
        if self.wpe.data is None:
            self._init_params(x)
        return self.forward(x)

    def forward(self, x):
        B, T = x.shape  # B是batch_size，T是序列长度
        xp = cuda.get_array_module(x)
        x = self.wte(x) + self.wpe[xp.arange(T)[None]]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class TransformerBlock(Layer):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.ln1 = L.LayerNorm()
        self.ln2 = L.LayerNorm()
        self.c_attn = L.Linear(in_size=n_embd, out_size=3 * n_embd)
        self.c_proj = L.Linear(in_size=n_embd, out_size=n_embd)
        self.fc1 = L.Linear(in_size=n_embd, out_size=4 * n_embd)
        self.fc2 = L.Linear(in_size=4 * n_embd, out_size=n_embd)

    def forward(self, x):
        B, T, C = x.shape
        xp = cuda.get_array_module(x)
        # ---- 1) LayerNorm ----
        x_norm = self.ln1(x)  # (B, T, C)

        # ---- 2) QKV projection ----
        qkv = self.c_attn(x_norm)  # (B, T, 3C)
        q, k, v = F.split(qkv, 3, axis=-1)  # each (B, T, C)

        # reshape into heads
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)

        # (B, heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # ---- 3) Scaled Dot-Product Attention ----
        att = F.matmul(q, k.transpose(0, 1, 3, 2))  # (B, heads, T, T)
        att = att / xp.sqrt(self.head_dim)

        # causal mask：只看前面位置
        mask = xp.tril(xp.ones((T, T), dtype=att.dtype))
        mask = mask[None, None]  # (1,1,T,T)
        mask = xp.where(mask == 1, 0, -1e9)

        att = att + mask
        att = F.softmax(att, axis=-1)
        out = F.matmul(att, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        att_out = self.c_proj(out)
        x = x + att_out
        x_norm = self.ln2(x)
        h = self.fc1(x_norm)
        h = F.gelu(h)
        h = self.fc2(h)
        x = x + h
        return x


# %%
hparams = {
    "n_vocab":30000,
    "n_seq":64,
    "n_embd":768,
    "hp_blocks" : [(768, 2), (768, 2)]
}
# hparams = {
#     "n_vocab": 30000,
#     "n_seq": 6,
#     "n_embd": 8,
#     "hp_blocks": [(8, 2), (8, 2)]
# }
# 创建 dataset
GPT2_dataset = GPT2SeqDataset(ids, seq_len=hparams["n_seq"])
# 创建 dataloader
GPT2_dataloader = GPTDataLoader(GPT2_dataset, batch_size=32, shuffle=True, gpu=True)
# 创建模型
model = GPT2(**hparams)
model.to_gpu()
optimizer = Adam()
optimizer.setup(model)
# %%
epochs = 20

for epoch in range(epochs):
    id = 0
    for x, t in GPT2_dataloader:
        id += 1
        y = model(x)  # (B, T, V)
        B, T, V = y.shape
        y2 = y.reshape(B * T, V)
        t2 = t.reshape(B * T)
        loss = F.softmax_cross_entropy(y2, t2)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        if id % 1000 == 0:
            print(epoch, id, loss.data)
    with open("text.json", "w", encoding="utf-8") as fp:
        json.dump([epoch], fp, ensure_ascii=False, indent=2)
model.save_weights("./model/pico-gpt.npz")

# 文本生成
model.load_weights("./model/pico-gpt.npz")   # 如果需要加载训练好的参数
model.to_gpu()

text = "三体舰队正在入侵！"
encoded = tokenizer.encode(text)
ids = encoded.ids

max_generate = 100   # 最多生成 100 个 token
n_seq = hparams["n_seq"]

xp = cuda.cupy

for _ in range(max_generate):
    # 取最近 n_seq 个 token 作为输入
    inp = ids[-n_seq:]
    inp = xp.array(inp, dtype=np.int32)[None]   # reshape => (1, T)

    # 模型输出 (1, T, vocab)
    logits = model(inp)
    logits = logits[0, -1]     # 取最后一个 token 的 logits, shape = (vocab,)

    # 采样（可改为 argmax）
    probs = F.softmax(logits, axis=-1).data
    next_id = int(xp.random.choice(len(probs), size=1, p=probs)[0])

    ids.append(next_id)

    # 若遇到 tokenizer 的 EOS id，可提前停止
    if next_id == tokenizer.token_to_id("<eos>"):
        break

# 解码
print("\n==== 生成文本 ====")
out_text = tokenizer.decode(ids)
print(out_text)
with open("text.json", "w", encoding="utf-8") as fp:
    json.dump([out_text], fp, ensure_ascii=False, indent=2)