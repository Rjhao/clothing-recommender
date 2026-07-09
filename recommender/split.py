"""统一的 leave-one-out 数据切分。

train.py 和 evaluate.py 共用此模块，消除两者对"最后一条交互"
理解不一致导致的 val/test 撞车：原先 2-interaction 用户的 items[-1]
既被 train.py 当 val、又被 evaluate.py 当 test，使 test item 进入
model selection（真实稀疏数据上会虚高 HR）。

切分规则（每用户按交互列表顺序，近似时序）：
    >=3 条: test=items[-1], val=items[-2], train=items[:-2]
    ==2 条: test=items[-1], train=items[0], 无 val
    ==1 条: 全 train, 无 val/test

负采样排除集应包含用户全部交互（train+val+test），见
build_user_interacted_set —— 避免 test item 被当训练负样本。
"""
from typing import Dict, List, Set, Tuple


def build_user_sequences(interactions, user2idx, item2idx) -> Dict[int, List[int]]:
    """按 user 聚合交互，保留列表顺序（近似时序）。

    Returns:
        {uid_idx: [item_idx, ...]}
    """
    user_seq: Dict[int, List[int]] = {}
    for inter in interactions:
        uid = int(inter['user_id'])
        iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_seq.setdefault(user2idx[uid], []).append(item2idx[iid])
    return user_seq


def leave_one_out_split(
    interactions,
    user2idx,
    item2idx,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Leave-one-out 切分。train.py 与 evaluate.py 共用。

    Returns:
        train_pairs: List[(uid_idx, item_idx)]
        val_pairs:   List[(uid_idx, item_idx)]
        test_pairs:  List[(uid_idx, item_idx)]
    """
    user_seq = build_user_sequences(interactions, user2idx, item2idx)

    train_pairs: List[Tuple[int, int]] = []
    val_pairs: List[Tuple[int, int]] = []
    test_pairs: List[Tuple[int, int]] = []

    for uid_idx, items in user_seq.items():
        if len(items) >= 3:
            for iid in items[:-2]:
                train_pairs.append((uid_idx, iid))
            val_pairs.append((uid_idx, items[-2]))
            test_pairs.append((uid_idx, items[-1]))
        elif len(items) == 2:
            # 2 条交互：最后一条 test，第一条 train，无 val
            train_pairs.append((uid_idx, items[0]))
            test_pairs.append((uid_idx, items[-1]))
        else:
            for iid in items:
                train_pairs.append((uid_idx, iid))

    return train_pairs, val_pairs, test_pairs


def build_user_interacted_set(
    interactions, user2idx, item2idx
) -> Dict[int, Set[int]]:
    """每用户全部交互商品集合（train+val+test），用于负采样排除。

    关键：排除集必须含 test item，否则 test item 会被采为训练负样本，
    教模型把测试物品排低（deflationary）。
    """
    user_items: Dict[int, Set[int]] = {}
    for inter in interactions:
        uid = int(inter['user_id'])
        iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], set()).add(item2idx[iid])
    return user_items


def build_user_train_count(
    interactions, user2idx, item2idx
) -> Dict[int, int]:
    """每用户训练交互数（用于 adaptive gate 的 interaction_counts）。

    用全部交互数近似（含 val/test 各 1 条），因 val/test 仅 1 条影响很小，
    且评估时用户的"历史交互数"本就含全部已知交互。
    """
    user_items = build_user_interacted_set(interactions, user2idx, item2idx)
    return {uid: len(items) for uid, items in user_items.items()}
