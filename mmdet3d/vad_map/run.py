"""独立注册和配置合成入口；复用原训练/测试脚本，不编辑任何原文件。"""
import argparse
import os
from pathlib import Path
import runpy
import sys
import tempfile


def plain(value):
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','test','config'])
    parser.add_argument('config')
    parser.add_argument('--base-config',help='可选：覆盖 YAML 中的完整单帧基础配置路径')
    parser.add_argument('--save-config',help='config 模式保存合成后的配置')
    args,extras=parser.parse_known_args()
    import yaml
    from torchpack.utils.config import Config
    from mmdet3d.utils import recursive_eval
    path=Path(args.config).resolve()
    overlay=yaml.safe_load(path.read_text())
    base=overlay.pop('base_config')
    base=Path(args.base_config).resolve() if args.base_config else (path.parent/base).resolve()
    cfg=Config(); cfg.load(str(base),recursive=True); cfg.update(overlay)
    if args.action=='config':
        cfg.update(extras)
        payload=yaml.safe_dump(plain(recursive_eval(dict(cfg))),sort_keys=False)
        if args.save_config: Path(args.save_config).write_text(payload)
        else: print(payload)
        return
    # 包已注册扩展；原脚本在同一进程执行，训练/验证子流程均能找到注册类型。
    root=Path(__file__).resolve().parents[2]
    script='train.py' if args.action=='train' else 'test.py'
    if args.action=='train' and ('LOCAL_RANK' in os.environ or '--launcher' in extras):
        script='dist_train.py'
    if 'LOCAL_RANK' in os.environ and '--launcher' not in extras:
        extras += ['--launcher','pytorch']
    with tempfile.TemporaryDirectory(prefix='daocc-vad-map-') as folder:
        effective=Path(folder)/'effective.yaml'
        effective.write_text(yaml.safe_dump(plain(cfg),sort_keys=False))
        sys.argv=[str(root/'tools'/script),str(effective)]+extras
        runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':
    main()
