"""Standalone sequential inference using the training-validation cache contract."""
import mmcv
import torch
from mmcv.parallel import scatter
from mmcv.runner import get_dist_info
from mmdet3d.core.hook.stream_eval_hooks import collect_stream_results


def stream_test(model, data_loader, tmpdir=None):
    """Run one stream per GPU; gather predictions by original dataset index.

    No DDP wrapper: ranks can own different numbers of complete sequences.
    Forward hooks on the local model can save predictions as frames finish.
    """
    modes = [(module, module.training) for module in model.modules()]
    rank, _ = get_dist_info()
    progress = mmcv.ProgressBar(len(data_loader)) if rank == 0 else None
    local_results = []
    try:
        model.eval()
        with model.validation_stream(), torch.no_grad():
            for batch in data_loader:
                data = scatter(batch, [torch.cuda.current_device()])[0]
                metas = data['stream_metas']
                if len(metas) != 1:
                    raise ValueError('Streaming evaluation requires one frame per rank')
                predictions = model(return_loss=False, rescale=True, **data)
                if len(predictions) != 1:
                    raise RuntimeError('Expected one prediction per frame')
                local_results.append((metas[0]['dataset_index'], predictions[0]))
                if progress is not None:
                    progress.update()
        return collect_stream_results(local_results, len(data_loader.dataset), tmpdir)
    finally:
        for module, training in modes:
            module.training = training
