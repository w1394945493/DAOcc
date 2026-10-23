"""注册高斯占据头，合成配置，复用原DAOcc训练/测试入口。"""
import argparse
from pathlib import Path
import runpy
import sys
import tempfile


def load_config(path):
    import yaml
    from torchpack.utils.config import Config
    path = Path(path).resolve()
    overlay = yaml.safe_load(path.read_text())
    if 'base_config' not in overlay:
        raise ValueError('高斯配置须包含base_config，指向原DAOcc完整配置')
    base = (path.parent / overlay.pop('base_config')).resolve()
    config = Config()
    config.load(str(base), recursive=True)
    config.update(overlay)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'test'])
    parser.add_argument('config')
    parser.add_argument('--daocc-root', help='已编译mmdet3d CUDA扩展的DAOcc源码目录')
    args, remaining = parser.parse_known_args()
    local_root = Path(__file__).resolve().parents[1]
    root = Path(args.daocc_root).resolve() if args.daocc_root else local_root
    #* 先固定本地models包，再选择底层mmdet3d，防止导入另一个DAOcc副本的models。
    sys.path.insert(0, str(local_root))
    import models
    sys.path.insert(0, str(root))
    from models.GaussDAOcc.head import GaussianOccHead
    import yaml
    config = load_config(args.config)
    # Torchpack按目录隐式继承；临时完整配置避免再次叠加原目录默认配置。
    with tempfile.TemporaryDirectory(prefix='gauss-daocc-config-') as directory:
        effective = Path(directory) / 'effective.yaml'
        effective.write_text(yaml.safe_dump(config.dict(), sort_keys=False))
        entry = root / 'tools' / (args.action + '.py')
        sys.argv = [str(entry), str(effective)] + remaining
        runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
