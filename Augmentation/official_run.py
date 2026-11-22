"""VisA 공식 분할(Dataset/visa/splits_official/{protocol})로 생성기를 돌린다 (11/20, 공개 수치 비교용).

공유 코드(sd_lora, ctransgan, pin_cleanup)는 고치지 않는다. 분할표 위치(common.kshot.SPLITS)만 공식 분할로 바꿔 끼운 뒤
원래 모듈을 그대로 __main__으로 실행한다. 공식 "불량 조금" 분할의 학습 불량 20장은 train_K20_seed0.csv에 있다.
ctransgan.generate_diverse는 --k 선택지가 1·2·4·8로 막혀 있어서, --k 8로 넘기고 run()에 들어갈 때 K를 20으로 바꾼다.

    cd Augmentation
    python official_run.py 2cls_fewshot sd_lora.train --k 20 --seed 0 --steps 1000 --p-normal 0.1 --rank 16 --runs-dir sd_lora/runs_official
    python official_run.py 2cls_fewshot ctransgan.generate_diverse --k 8 --seed 0 --diverse-scratch --device cuda --out ...
"""
import runpy
import sys
from pathlib import Path

import common.kshot as kshot

REAL_K = 20


def main():
    protocol, module, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    splits = kshot.ROOT / 'Dataset/visa/splits_official' / protocol
    assert (splits / f'train_K{REAL_K}_seed0.csv').exists(), splits
    kshot.SPLITS = splits
    if module == 'ctransgan.generate_diverse':
        import ctransgan.generate_diverse as gd
        orig = gd.run
        gd.run = lambda k, seed, a: orig(REAL_K, seed, a)
        if '--splits' not in args:
            args += ['--splits', str(splits)]
        sys.argv = [module] + args
        gd.main()
        return
    sys.argv = [module] + args
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    print('분할표:', Path(kshot.ROOT / 'Dataset/visa/splits_official' / sys.argv[1]), flush=True)
    main()
