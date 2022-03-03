import re
from collections import Counter, defaultdict, namedtuple
from difflib import get_close_matches

import neologdn
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

Item = namedtuple("Item", ["repr", "id"])
NUM_HOLDS = 5
CUTOFF = 0.85
# よくある部分文字列を削除（approx matchでの誤ヒットを防ぐ、長いものほど効果的）
prepro_re = re.compile(r"pharmaceuticals?|biotherapeutics?|therapeutics?|laboratories|biosciences?")
with_gold = True


ID_KEY = "ID"
INPUT_KEY = "入力文字列名"
REPR_KEY = "正規化文字列"
LABEL_KEY = "クラスタラベル"
csv_path = "/app/workspace/input.csv"
output_path = "/app/workspace/input_cluster.csv"

# TODO: 短いエントリをフィルタ

def normalize(s):
    return neologdn.normalize(re.sub(r"\s+", " ", s.lower()))


# Load data
df = pd.read_csv(csv_path)
df = df.set_index(ID_KEY)
print("欠損値の除去による行数変化")
print(df.shape, df.dropna(how="all").shape[0])
_df = df[df.dropna(how="all", axis=1).head().columns]  # 列
__df = _df.dropna(how="all").copy()  # 行
# Normalize text
__df.loc[:, REPR_KEY] = __df[INPUT_KEY].apply(normalize).values

# Prepare data
items = [
    Item(prepro_re.sub("", w), i)
    for i, w in __df[REPR_KEY].items()
    if isinstance(w, str)
]
repr2id = {i.repr: i.id for i in items}
id2repr = {i.id: i.repr for i in items}
repr2enum = {k: ix for ix, (k, v) in enumerate(repr2id.items())}
id2enum = {v: ix for ix, (k, v) in enumerate(repr2id.items())}
enum2id = {v: k for k, v in id2enum.items()}


# Calc Similarities
reprs = sorted(list(repr2id.keys()))
r2similars = {r: get_close_matches(r, reprs, n=NUM_HOLDS, cutoff=CUTOFF) for r in reprs}
print(f"閾値{CUTOFF}に対する、類似語群のサイズと頻度（最大数:{NUM_HOLDS}）:")
print(Counter(len(v) for k, v in r2similars.items()).most_common())

# Calc Clusters
N = len(r2similars)
i, j = [], []
for r, sims in r2similars.items():
    keyix = repr2enum[r]
    valixs = [repr2enum[s] for s in sims if s != r]  #
    for valix in valixs:
        i.append(keyix)
        j.append(valix)
d = [1 for _ in i]
csr = csr_matrix((d, (i, j)), (N, N))
n, labels = connected_components(csr, connection="strong", directed=False)

indexes = [enum2id[ix] for ix, cid in enumerate(labels)]
values = [cid for ix, cid in enumerate(labels)]
__df.loc[indexes , LABEL_KEY] = values
print("種々の原因でラベルがつかないものの数:", __df[__df[LABEL_KEY].isnull()].shape[0])


# 入力が異なるが正規化文字列が同じものにラベル伝搬させる処理
# NOTE: 元の入力が重複していた場合消えてしまう
input2indexs = defaultdict(list)
for ix, v in __df[INPUT_KEY].items():
    input2indexs[v].append(ix)
repr2inputs = defaultdict(list)
for _, (v, k) in __df[[INPUT_KEY, REPR_KEY]].iterrows():
    repr2inputs[k].append(v)
repr2inputs_m = {k:v for k,v in repr2inputs.items() if len(v) > 1}
indices, fills = [], []
for ix, r in __df[__df[LABEL_KEY].isnull()][REPR_KEY].items():
    original_ixs = [ix for i in repr2inputs[r] for ix in input2indexs[i]]
    _d = dict(__df.loc[original_ixs][LABEL_KEY].items())
    ls = [v for v in _d.values() if not np.isnan(v)]
    if len(ls) > 0:
        l = ls[0]
        for k, v in _d.items():
            if np.isnan(v):
                indices.append(k)
                fills.append(l)
    else:
        print("FILLING FAILED: ", ix, r)  # 原因不明のNaN
___df = __df.copy()
___df.loc[indices , LABEL_KEY] = fills
# 原因不明な広いこぼしはfillna(-1)しとく
print("原因不明なエラー:")
print(___df[___df[LABEL_KEY].isnull()].shape[0])
print(___df[___df[LABEL_KEY].isnull()].index)
___df = ___df.fillna(value={LABEL_KEY:-1})


# Export
___df.to_csv(output_path)

print()
# Compare Results
if with_gold:
    from pprint import PrettyPrinter

    pp = PrettyPrinter()

    GOLD_KEY = "正式当事者名1"

    print("====EVALUATION====")
    id2cid = {enum2id[ix]: cid for ix, cid in enumerate(labels)}
    cid2ids = defaultdict(list)
    for _id, cid in id2cid.items():
        cid2ids[cid].append(_id)
    repr2gold = {
        prepro_re.sub("", k): v
        for _, (k, v) in __df[[REPR_KEY, GOLD_KEY]].iterrows()
        if isinstance(k, str)
    }
    gold2reprs = defaultdict(set)
    for _, (v, k) in __df[[REPR_KEY, GOLD_KEY]].iterrows():
        if isinstance(v, str):
            gold2reprs[k].add(prepro_re.sub("", v))

    # pred -> gold 方向の比較
    print("予測に対する正解群:")
    pred_golds = []
    for cid, ids in cid2ids.items():
        reprs = [id2repr[_id] for _id in ids]
        golds = {repr2gold[r] for r in reprs}
        gold_reprs = [gold2reprs[g] for g in golds]
        pred_golds.append((reprs, gold_reprs))

    for p, gs in pred_golds[:100]:
        if len(p) > 1:
            print('PREDICTION:')
            pp.pprint(p)
            print('GOLDs:')
            pp.pprint(gs)
            print()

    # gold->preds 方向の比較
    print("正解に対する予測群:")
    repr2cid = {r: labels[ix] for r, ix in repr2enum.items()}
    gold_preds = []
    for gold, _reprs in gold2reprs.items():
        for _repr in _reprs:
            cid = repr2cid[_repr]
            ids = cid2ids[cid]
            preds = [id2repr[_id] for _id in ids]
            gold_preds.append((_reprs, preds))

    golds2preds = {tuple(gs): ps for gs, ps in gold_preds}
    from pprint import PrettyPrinter

    pp = PrettyPrinter()
    for gs, ps in list(golds2preds.items())[:100]:
        if len(gs) > 1:
            print('GOLD:')
            pp.pprint(gs)
            print('PREDICTIONs:')
            pp.pprint(ps)
            print()