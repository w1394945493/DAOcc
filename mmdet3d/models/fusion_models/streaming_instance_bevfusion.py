"""流式历史 BEV 融合与当前帧实例注入的组合模型。"""
from mmdet3d.models.builder import FUSIONMODELS
from .streaming_bevfusion import StreamingBEVFusion
from .instance_bevfusion import InstanceBEVFusion


@FUSIONMODELS.register_module()
class StreamingInstanceBEVFusion(StreamingBEVFusion, InstanceBEVFusion):
    #* (11.17) 按 MRO 复用两个父类，不复制编码器、缓存和预测代码。
    # 初始化：StreamingBEVFusion → InstanceBEVFusion → BEVFusion。
    # 前向：流式父类管理元数据/缓存，再通过 super() 调用实例父类的 forward_single。
    # 实例父类在 decoder 前/后调用 process_bev_features，解析到流式父类进行历史融合；
    # 随后 forward_heads 提取当前检测实例并注入占据 BEV，最终由流式父类更新缓存。
    # 缓存仍是所选阶段的融合 BEV，不缓存实例，也不把实例注入后的占据 BEV 写回缓存。
    # 流式验证的缓存隔离、首帧和序列切换规则全部继承 StreamingBEVFusion。
    pass
