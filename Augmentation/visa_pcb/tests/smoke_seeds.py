import itertools
import numpy as np

from src.seeds import (METHOD_IDS, PCB_IDS, K_VALUES, SPLIT_SEEDS,
                       condition_seed, sample_rng)
from tests.smoke_stage1 import must_fail


def run():
    combos = list(itertools.product(METHOD_IDS, PCB_IDS, K_VALUES, SPLIT_SEEDS))
    seeds = [condition_seed(*c) for c in combos]
    assert len(set(seeds)) == len(combos), 'condition seeds collide'
    assert not set(seeds) & {*SPLIT_SEEDS, 7017}, 'generation seed equals a raw split seed'
    assert all(0 <= s < 2**63 for s in seeds)
    assert condition_seed('copypaste', 'pcb1', 2, 0) == condition_seed('copypaste', 'pcb1', 2, 0)

    # AD 팀이 seed s로 만든 난수열과 생성 난수열이 다르다
    for s in SPLIT_SEEDS:
        ad = np.random.default_rng(s).random(8)
        ours = sample_rng('copypaste', 'pcb1', 1, s, 0).random(8)
        assert not np.allclose(ad, ours)
    a = sample_rng('copypaste', 'pcb1', 1, 0, 0).random(4)
    b = sample_rng('copypaste', 'pcb1', 1, 0, 1).random(4)
    assert not np.allclose(a, b)

    must_fail(lambda: condition_seed('copypaste', 'pcb4', 1, 0))
    must_fail(lambda: condition_seed('copypaste', 'pcb1', 3, 0))
    must_fail(lambda: condition_seed('copypaste', 'pcb1', 1, 5))
    return {'status': 'PASS', 'conditions': len(combos)}


if __name__ == '__main__':
    print(run())
