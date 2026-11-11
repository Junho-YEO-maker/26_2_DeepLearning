"""환경 호환 처리.

Colab에는 torchao 0.10이 미리 깔려 있는데, 최신 peft는 LoRA를 만들 때 torchao 버전을 확인하고
0.16보다 낮으면 ImportError로 멈춘다. 우리는 torchao를 쓰지 않으므로, 오래된 torchao가 있으면
peft가 "torchao 없음"으로 보게 한다. (노트북에서 지우지 못한 경우에도 학습이 돌게 하는 안전장치)
"""


def disable_old_torchao(min_version='0.16.0'):
    try:
        from importlib.metadata import version, PackageNotFoundError
        from packaging.version import Version
        try:
            if Version(version('torchao')) >= Version(min_version):
                return False
        except PackageNotFoundError:
            return False
    except ImportError:
        return False
    import importlib
    off = lambda *a, **k: False
    for name in ('peft.import_utils', 'peft.tuners.lora.torchao', 'peft.tuners.lora.model'):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, 'is_torchao_available'):
            mod.is_torchao_available = off
    print('[호환] 오래된 torchao가 있어 peft에서 쓰지 않게 했습니다', flush=True)
    return True
