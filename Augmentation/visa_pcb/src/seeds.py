"""K장 실험의 생성 난수 시드.

AD 팀의 split seed s(0–4)는 "어떤 불량 K장을 뽑았나"만 뜻한다.
Augmentation 팀은 s를 난수 시드로 직접 쓰지 않고, 팀 고유 값(AUG_NAMESPACE)과
(방법, PCB, K, s)를 함께 넣어 생성 시드를 파생한다. 그래서 AD 팀이 seed 0으로
뽑은 난수열과 생성 난수열이 겹치지 않고, 조건끼리도 서로 다른 난수열을 쓴다.
"""
import numpy as np

AUG_NAMESPACE = 7017
METHOD_IDS = {'copypaste': 1, 'ctransgan': 2, 'sd_lora': 3, 'classic': 4}
PCB_IDS = {'pcb1': 1, 'pcb2': 2, 'pcb3': 3}
SPLIT_SEEDS = range(5)
K_VALUES = (1, 2, 4, 8, 16, 24)


def _entropy(method, pcb, k, split_seed):
    assert method in METHOD_IDS, f'unknown method: {method}'
    assert pcb in PCB_IDS, f'unknown pcb: {pcb}'
    assert k in K_VALUES, f'unknown K: {k}'
    assert split_seed in SPLIT_SEEDS, f'split seed must be 0-4: {split_seed}'
    return [AUG_NAMESPACE, METHOD_IDS[method], PCB_IDS[pcb], k, split_seed]


def condition_seed_sequence(method, pcb, k, split_seed):
    """조건 하나(방법, PCB, K, s)의 numpy SeedSequence."""
    return np.random.SeedSequence(_entropy(method, pcb, k, split_seed))


def condition_seed(method, pcb, k, split_seed):
    """torch.manual_seed 등에 넣을 63비트 정수 시드. manifest의 gen_seed 열에 기록한다."""
    state = condition_seed_sequence(method, pcb, k, split_seed).generate_state(2, np.uint32)
    return (int(state[0]) << 31) ^ int(state[1])


def sample_rng(method, pcb, k, split_seed, index):
    """생성 샘플 index번의 numpy Generator."""
    return np.random.default_rng(np.random.SeedSequence(_entropy(method, pcb, k, split_seed) + [index]))
