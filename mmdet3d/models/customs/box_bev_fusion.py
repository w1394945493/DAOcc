"""预测框引导的实例→占据 BEV 融合；不读取 GT 或预测占据类别。"""
import math
import torch
from torch import nn
from mmdet3d.models.builder import CUSTOMS


@CUSTOMS.register_module()
class BoxBEVFusion(nn.Module):
    def __init__(self, instance_channels=64, bev_channels=128, num_classes=10,
                 max_instances=100, gate_init=0.01,
                 edge_matching=False, local_edges=False,
                 aggregation='weighted_mean', attention_channels=32):
        super().__init__()
        if max_instances < 1 or not 0 < gate_init < 1:
            raise ValueError('max_instances须为正，gate_init须在(0,1)内')
        if aggregation not in ('weighted_mean', 'attention'):
            raise ValueError("aggregation须为'weighted_mean'或'attention'")
        if not isinstance(attention_channels, int) or attention_channels < 1:
            raise ValueError('attention_channels须为正整数')
        self.max_instances = max_instances
        self.bev_channels = bev_channels
        self.aggregation = aggregation
        #* (11.19) 独立开关：旧配置不改变；新配置显式开启逐边匹配和局部枚举。
        self.local_edges = local_edges
        self.edge_match = None
        if edge_matching:
            # 先加 LayerNorm，主要是因为输入拼接了三种数值尺度可能不同的信息：实例特征：经过线性投影，并加上类别编码；当前占据特征：来自占据 BEV；几何信息：包含归一化位置、对数尺寸、正余弦及检测分数。
            # LayerNorm(265) 对每条边自己的 265 维输入进行归一化，避免匹配网络过度受某部分输入幅值影响。它不会在不同边之间求平均，也不依赖 batch 大小。
            self.edge_match = nn.Sequential(
                nn.LayerNorm(2 * bev_channels + 9),
                nn.Linear(2 * bev_channels + 9, 32), nn.ReLU(), nn.Linear(32, 1))
            # 初始 sigmoid=0.5，只保守接收消息；不是经过标定的检测置信度。
            nn.init.zeros_(self.edge_match[-1].weight)
            nn.init.zeros_(self.edge_match[-1].bias)
        self.content = nn.Linear(instance_channels, bev_channels)
        self.category = nn.Embedding(num_classes, bev_channels)
        # 局部归一化 XY(2)、中心高度(1)、log尺寸(3)、sin/cos(2)、置信度(1)。
        self.geometry = nn.Sequential(nn.Linear(9, bev_channels), nn.ReLU(), nn.Linear(bev_channels, bev_channels))
        self.message = nn.Sequential(nn.LayerNorm(bev_channels), nn.ReLU(), nn.Linear(bev_channels, bev_channels))
        # 可选轻量单头注意力：占据特征生成 Query，逐边实例消息生成 Key。
        # 默认 weighted_mean 不创建这两个投影，保持原参数量、初始化和计算路径不变。
        if aggregation == 'attention':
            self.attention_query = nn.Linear(bev_channels, attention_channels, bias=False)
            self.attention_key = nn.Linear(bev_channels, attention_channels, bias=False)
            self.attention_scale = attention_channels ** -0.5
        else:
            self.attention_query = None
            self.attention_key = None
            self.attention_scale = None
        self.gate = nn.Linear(bev_channels * 2, bev_channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    @staticmethod
    @torch.no_grad()
    def query_points(sampler, lidar_aug_matrix, lidar2ego, occ_aug_matrix):
        """与原 CrossCoordinateSample 的 ego 分支一致，返回 [B,Y*X,3] 米制坐标。

        保留原实现只对 OCC 增强取3×3的约定；不修改原采样器及其查询点。
        """
        if sampler.point_type != 'ego' or sampler.extra_up:
            raise ValueError('首版融合仅支持无 extra_up 的 ego CrossCoordinateSample')
        with torch.cuda.amp.autocast(enabled=False):
            aug, ext, occ = lidar_aug_matrix.float(), lidar2ego.float(), occ_aug_matrix.float()
            # C1. 对应可视化 occ_to_lidar：ego增强网格 → 撤销OCC增强 → 原ego
            # → inverse(lidar2ego) → 原LiDAR → 应用LiDAR增强。每个样本使用自己的矩阵。
            # 行向量形式的右乘R^T等价于列向量左乘R；所有位置均为米制浮点坐标。
            points = sampler.ref_points.detach().float()[None].expand(aug.shape[0], -1, -1)
            ego = points @ torch.linalg.inv(occ[:, :3, :3]).transpose(1, 2)
            lidar = (ego - ext[:, None, :3, 3]) @ torch.linalg.inv(ext[:, :3, :3]).transpose(1, 2)
            return lidar @ aug[:, :3, :3].transpose(1, 2) + aug[:, None, :3, 3]

    @torch.no_grad()
    def build_edges(self, points, instances, source_bounds, grid_shape=None):
        """分块建边，内存不构造 N×M×C；无交集的框不占实例数量配额。"""
        
        #*----------------------------------------------------------------------------#
        #* (11.19) forward 提供真实网格形状；外部旧调用不传形状时仍走原全网格路径。
        if self.local_edges and grid_shape is not None:
            return self._local_edges(points, instances, source_bounds, grid_shape)
        
        # C2. 对应可视化 make_edges(..., use_height=False)：单个样本内按框建平面连接。
        # points和boxes均在增强后LiDAR系；连接目标仍保留原ego占据网格的展平编号。
        boxes, scores = instances['boxes'].detach().float(), instances['scores'].detach().float()
        xmin, ymin, xmax, ymax = source_bounds
        valid = torch.isfinite(points).all(-1)
        valid &= (points[:, 0] >= xmin) & (points[:, 0] < xmax)
        valid &= (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        #* (11.16) 优化前建边实现保留如下（仅注释，不执行）。
        # 耗时原因：每个框分别扫描全部网格，并分别启动多个 CUDA 运算。
        # 对 CUDA Tensor 执行 Python 的 if/not 判断需要取得布尔值，会触发 CPU 等待 GPU；
        # 每框 nonzero 还会产生动态长度输出，带来同步与调度开销。
        # 原实现遇到足够多有效框会提前停止；新版分块计算全部候选关系，减少同步，
        # 代价是额外的布尔关系矩阵，因此最终收益仍以实际 GPU 测速为准。
        # -------------------- 原逐框实现开始 --------------------
        # sources, targets, geometry = [], [], []
        # used = 0
        # for i in scores.argsort(descending=True):
        #     box = boxes[i]
        #     if not torch.isfinite(box).all() or not (box[3:6] > 0).all() or not torch.isfinite(scores[i]):
        #         continue
        #     # 平移到框中心，再旋转-yaw，转为框局部XY；在局部系判断矩形包含关系。
        #     # 不是用轴对齐包围框近似，也不是将网格下标直接与米制框参数比较。
        #     delta = points[:, :2] - box[:2]
        #     c, s = box[6].cos(), box[6].sin()
        #     local = torch.stack([c * delta[:, 0] + s * delta[:, 1],
        #                          -s * delta[:, 0] + c * delta[:, 1]], -1)
        #     # BEV单元代表高度柱，故只检查XY。查询平面z=0不必落在框的高度区间内；
        #     # z/高度随后仅进入几何编码，这不同于可视化中完整3D框与GT体素的包含检查。
        #     mask = valid & (local.abs() <= box[3:5] / 2).all(-1)
        #     target = mask.nonzero(as_tuple=True)[0]
        #     if not target.numel():
        #         continue
        #     # 每条边有不同的相对位置；除以框尺寸后，框内局部XY大致在[-0.5,0.5]。
        #     # 其余7维同框共享：z/10、log(dx,dy,dz)、sin/cos(yaw)、分类置信度。
        #     xy = local[target] / box[3:5].clamp_min(1e-3)
        #     attr = torch.cat([box[2:3] / 10, box[3:6].clamp_min(1e-3).log(),
        #                       box[6:7].sin(), box[6:7].cos(), scores[i:i+1]])
        #     # 例如实例i覆盖单元[10,11]，就产生边(i,10)、(i,11)。
        #     # 返回的src/dst均为[E]，等价于脚本instance_to_occ_bev的两行；geo为[E,9]。
        #     # 重叠框可重复指向同一单元，留给下面聚合，不在这里强行选唯一实例。
        #     sources.append(i.expand_as(target)); targets.append(target)
        #     geometry.append(torch.cat([xy, attr[None].expand(len(target), -1)], -1))
        #     used += 1
        #     if used >= self.max_instances:
        #         break
        # if not targets:
        #     empty = torch.empty(0, dtype=torch.long, device=points.device)
        #     return empty, empty, points.new_empty((0, 9))
        # return torch.cat(sources), torch.cat(targets), torch.cat(geometry)
        # -------------------- 原逐框实现结束 --------------------

        #* (11.16) 分块批量判断框内关系，避免逐框 Python 判断引发 GPU/CPU 同步。
        # 只保留 N×M 布尔关系；临时浮点坐标每块最多32个框，不构造 N×M×C 特征。
        order = scores.argsort(descending=True)
        ordered = boxes[order]
        good = torch.isfinite(ordered).all(-1) & (ordered[:, 3:6] > 0).all(-1)
        good &= torch.isfinite(scores[order])
        masks = []
        for start in range(0, len(order), 32):
            box = ordered[start:start + 32]
            delta_x = points[None, :, 0] - box[:, None, 0]
            delta_y = points[None, :, 1] - box[:, None, 1]
            c, sn = box[:, None, 6].cos(), box[:, None, 6].sin()
            local_x = c * delta_x + sn * delta_y
            local_y = -sn * delta_x + c * delta_y
            inside = (local_x.abs() <= box[:, None, 3] / 2)
            inside &= (local_y.abs() <= box[:, None, 4] / 2)
            inside &= valid[None] & good[start:start + 32, None]
            masks.append(inside)
        if not masks:
            empty = torch.empty(0, dtype=torch.long, device=points.device)
            return empty, empty, points.new_empty((0, 9))
        inside = torch.cat(masks, dim=0)
        # 按分数顺序选前 max_instances 个有交集实例，空框仍不占配额。
        has_edges = inside.any(dim=1)
        keep = has_edges & (has_edges.long().cumsum(0) <= self.max_instances)
        inside &= keep[:, None]
        rank, dst = inside.nonzero(as_tuple=True)
        src = order[rank]
        # nonzero 按行输出，与原来的“逐框、逐网格”边顺序一致。
        box = boxes[src]
        # 每条边对应一对“预测框 src -> OCC 网格 dst”。同一个框可对应多个 dst，
        # 因而下面的 delta 会为该框覆盖的每个 OCC 网格分别计算相对位置。
        delta = points[dst, :2] - box[:, :2]
        c, sn = box[:, 6].cos(), box[:, 6].sin()
        # 将“网格相对框中心的 XY 偏移”旋转到框自身的局部坐标系。
        #! 即使 src 相同，只要 dst 不同，delta 和 local 就不同；这正是框与网格的几何关系。
        local = torch.stack([c * delta[:, 0] + sn * delta[:, 1], -sn * delta[:, 0] + c * delta[:, 1]], dim=-1)
        # 用框的长宽归一化，得到框内相对位置 local_xy；通常框中心附近约为 (0, 0)，
        # 框边缘附近约为 +/-0.5。它不只是框自身的几何属性。
        xy = local / box[:, 3:5].clamp_min(1e-3)
        #! 同一个框覆盖的不同 OCC 网格具有不同的 xy，因此 geo 不同，后续会生成不同的几何消息。
        geo = torch.cat([xy, box[:, 2:3] / 10, box[:, 3:6].clamp_min(1e-3).log(),
                         sn[:, None], c[:, None], scores[src, None]], dim=-1)
        return src, dst, geo

    @torch.no_grad()
    def _local_edges(self, points, instances, source_bounds, grid_shape):
        """规则 OCC 平面经仿射变换后，在网格索引空间局部枚举，再精确检查旋转框。"""
        height, width = grid_shape
        # 单行/单列没有可逆二维基底，复用原判断。
        if min(height, width) < 2:
            return self.build_edges(points, instances, source_bounds)
        boxes, scores = instances['boxes'].detach().float(), instances['scores'].detach().float()
        order = scores.argsort(descending=True)
        boxes_sorted = boxes[order]
        good = torch.isfinite(boxes_sorted).all(-1) & (boxes_sorted[:, 3:6] > 0).all(-1)
        good &= torch.isfinite(scores[order])
        # 不让非法框进入坐标转换和整数化；对应枚举长度随后置0。
        safe = torch.where(good[:, None], boxes_sorted, torch.zeros_like(boxes_sorted))
        origin = points[0, :2]
        basis = torch.stack([points[1, :2] - origin, points[width, :2] - origin])
        # 基底分别对应列、行步长，支持非方形网格、旋转、翻转、缩放和倾斜投影。
        if not torch.isfinite(basis).all() or basis.det().abs() < 1e-8:
            return self.build_edges(points, instances, source_bounds)
        offsets = safe.new_tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        local = safe[:, None, 3:5] * offsets
        c, sn = safe[:, 6].cos()[:, None], safe[:, 6].sin()[:, None]
        corners = torch.stack([c * local[..., 0] - sn * local[..., 1],
                               sn * local[..., 0] + c * local[..., 1]], -1) + safe[:, None, :2]
        ij = (corners - origin) @ torch.linalg.inv(basis)
        # 多留一格消除浮点边界舍入漏边；最终仍用原旋转框条件精确裁掉。
        low = (ij.amin(1).floor() - 1).clamp(min=0)
        high = ij.amax(1).ceil() + 1
        limits = ij.new_tensor([width - 1, height - 1])
        low = torch.minimum(low, limits + 1).long()
        high = torch.minimum(high, limits).clamp(min=-1).long()
        sizes = (high - low + 1).clamp_min(0)
        counts = sizes.prod(-1) * good.long()
        # 一次批量生成变长局部区域，避免逐框 Python/CUDA 同步；不构造 N×全网格掩码。
        rank = torch.repeat_interleave(torch.arange(len(order), device=points.device), counts)
        starts = counts.cumsum(0) - counts
        offset = torch.arange(rank.numel(), device=points.device) - starts[rank]
        cols = low[rank, 0] + offset % sizes[rank, 0].clamp_min(1)
        rows = low[rank, 1] + offset // sizes[rank, 0].clamp_min(1)
        dst = rows * width + cols
        box = boxes_sorted[rank]
        # rank 标识候选框，dst 标识该框覆盖的具体 OCC 网格；一个 rank 通常对应多个 dst。
        delta = points[dst, :2] - box[:, :2]
        c, sn = box[:, 6].cos(), box[:, 6].sin()
        # 将每个 OCC 网格相对框中心的偏移旋转到对应框的局部坐标系。
        #! 同一框内不同网格的 dst 不同，所以这里得到的 xy 也不同。
        xy = torch.stack([c * delta[:, 0] + sn * delta[:, 1],
                          -sn * delta[:, 0] + c * delta[:, 1]], -1)
        xmin, ymin, xmax, ymax = source_bounds
        inside = (xy.abs() <= box[:, 3:5] / 2).all(-1)
        inside &= torch.isfinite(points[dst]).all(-1)
        inside &= (points[dst, 0] >= xmin) & (points[dst, 0] < xmax)
        inside &= (points[dst, 1] >= ymin) & (points[dst, 1] < ymax)
        rank, dst, xy = rank[inside], dst[inside], xy[inside]
        # 与旧实现一致：按分数取有交集的前 max_instances 个框，而非先截断候选。
        occupied = torch.bincount(rank, minlength=len(order)) > 0
        keep = occupied.long().cumsum(0)[rank] <= self.max_instances
        rank, dst, xy = rank[keep], dst[keep], xy[keep]
        src = order[rank]
        box = boxes[src]
        # 前两维是归一化 local_xy，描述“框 -> 当前 OCC 网格”的相对几何；
        # 后七维描述框的高度、尺寸、朝向和置信度。
        #! local_xy 随 dst 变化，使同一实例发往不同 OCC 网格的 geo 和几何消息各不相同。
        geo = torch.cat([xy / box[:, 3:5].clamp_min(1e-3), box[:, 2:3] / 10,
                         box[:, 3:6].clamp_min(1e-3).log(), box[:, 6:7].sin(),
                         box[:, 6:7].cos(), scores[src, None]], -1)
        return src, dst, geo

    def forward(self, bev, instances, points, source_bounds):
        """每个样本独立聚合；未关联单元的更新严格为0，返回原 dtype/形状。"""
        bsz, channels, height, width = bev.shape
        if len(instances) != bsz or points.shape != (bsz, height * width, 3) or channels != self.bev_channels:
            raise ValueError('实例列表、查询网格与占据 BEV 形状不一致')
        outputs = []
        # FP32 几何、消息和累加防止大量边在混合精度训练中溢出。
        with torch.cuda.amp.autocast(enabled=False):
            #*=========================================================================#
            # 无有效边时会跳过消息/门控分支；若直接返回 BEV，融合参数可能未参与计算，grad 为 None。
            # 每个参数 Tensor 取一个元素乘0，构造数值为0、但依赖全部融合参数的标量。
            # 下方将它加到输出后，反向会为这些参数产生零梯度，而非将其视作 unused parameter，
            # 避免本模块的空候选分支触发 DDP(find_unused_parameters=False) 的未使用参数问题。
            #! (11.23) 即使整个 batch 没有选中任何预测框，多卡训练也不会因融合参数未使用而报错。
            # 这里只补计算图依赖；有有效边时，正常融合路径仍可贡献非零梯度。
            zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters())
            for b in range(bsz):
                flat = bev[b].float().permute(1, 2, 0).reshape(-1, channels) # (40000 128)
                item = instances[b] # 第b个样本的全部候选实例组成的字典
                #*-----------------------------------------------------------------------------#
                # 构建边邻接关系：points[b]：第 b 个样本中，每个占据 BEV 单元对应的空间查询点坐标，
                src, dst, geo = self.build_edges(points[b].detach().float(), item, source_bounds, (height, width))
                #! 空实例或实例未覆盖有效 OCC 网格时 src 为空，下面的实例消息、聚合和门控全部跳过；
                #! flat 保持原 OCC 特征，仍会继续进入占据预测头并正常计算 OCC loss。
                # D1. 每次只处理本样本的局部索引，无跨batch连接；flat为[Yocc*Xocc,C]。
                if src.numel():
                    #*-------------------------------------------------------------------------#
                    # 为每个“实例→单元”生成不同的消息： 
                    # D2. 64维来源特征投影到128维，并加入检测类别embedding。
                    # content[src]把同一实例复制到它的各条边，再加每条边的几何编码形成消息。
                    content = self.content(item['features'].float()) + self.category(item['labels'])
                    # 消息仍编码实例与相对几何；新增匹配分支在聚合之前参考目标单元。
                    # content[src] 对同一个框相同，但 geo 中的 local_xy 随目标网格 dst 改变。
                    #! 因此同一个框覆盖的不同 OCC 网格会得到不同的 geometry(geo)，
                    #! 最终生成不同的 messages，而不是把同一个实例向量原样广播到整个框内。
                    messages = self.message(content[src] + self.geometry(geo)) # 消息包含：检测来源网格提取的实例特征；检测类别 embedding；实例与目标单元之间的几何信息。
                    score_prior = item['scores'].detach().float()[src].clamp(0, 1)[:, None]

                    #*--------------------------------------------------------------------------#
                    #* (11.19) 每条边联合实例、目标占据单元和几何，学习接收程度。
                    #! 不可行：实例框过大时，可能关联多个目标占据单元，导致计算开销显著增加
                    # 检测分数作为先验；sigmoid 不强迫所有实例权重和为1，允许都不接收。
                    # 后续保留原归一化与总权重幅度控制，避免单边时匹配权重被完全抵消。
                    if self.edge_match is not None:
                        #*----------------------------------------------------------------------#
                        # 实例内容 content[i]     [128]  实例特征投影 + 类别编码  
                        # 目标单元 flat[j]        [128]  尚未注入的占据 BEV 特征
                        # 相对几何 geo[i,j]       [9]    相对位置、框尺寸、朝向等
                        # 同一个实例连接不同单元时，目标特征和相对位置不同，匹配权重也可以不同。
                        match = self.edge_match(torch.cat([content[src], flat[dst], geo], -1)).sigmoid()
                        score_prior = score_prior * match
                    # D3. index_add按dst将消息汇入目标BEV；重叠实例按置信度归一化，
                    # 避免仅因连接数量更多导致消息幅值线性增大。未关联单元total=denom=0。
                    #* (11.16) 优化前全网格聚合/门控实现保留如下（仅注释，不执行）。
                    # 耗时原因：为全部网格分配并计算聚合结果，且对所有网格执行 gate Linear。
                    # 当前200×200网格共40000个位置，即使没有实例关联也计算256→128的门控。
                    # 无边位置 averaged=0，门控结果乘0，不影响输出；新版仅处理实际关联位置。
                    # 下列实现是可避免的计算开销，不代表此前557ms全部来自这一段。
                    # -------------------- 原全网格实现开始 --------------------
                    # total = torch.zeros_like(flat).index_add(0, dst, messages * weights)
                    # denom = flat.new_zeros((len(flat), 1)).index_add(0, dst, weights)
                    # averaged = total / denom.clamp_min(1e-6)
                    # # D4. 每个目标单元结合原特征与聚合消息，预测逐通道门控（非注意力softmax）。
                    # # 初始门控约gate_init；残差保留原占据信息。无边单元averaged=0，更新严格为0。
                    # gate = torch.sigmoid(self.gate(torch.cat([flat, averaged], -1)))
                    # # 小置信度实例不仅影响重叠归一化，也会降低整体注入幅度。
                    # flat = flat + gate * averaged * denom.clamp(max=1)
                    # -------------------- 原全网格实现结束 --------------------

                    #* (11.16) 只在实际关联单元上聚合和计算门控，再散射回原网格。
                    # 无边位置原更新严格为0，无需执行128通道门控 Linear。
                    active, inverse = torch.unique(dst, sorted=True, return_inverse=True)
                    if self.aggregation == 'attention':
                        # src[e]、dst[e] 描述第 e 条“实例 -> OCC 单元”边：
                        # messages[e] 是这条边携带的实例消息，flat[dst[e]] 是接收它的 OCC 特征。
                        # 如果两个实例都覆盖同一个 OCC 单元，它们的 dst 相同，但 messages 不同。

                        # 每条边都用目标 OCC 特征生成 Query，用该边的实例消息生成 Key。
                        # flat[dst] 会按照边表重复取 OCC 特征，因此 query/key 均为 [边数 E, A]；
                        # A=attention_channels。相同目标 OCC 的 Query 相同，不同实例的 Key 不同。
                        query = self.attention_query(flat[dst])
                        key = self.attention_key(messages)

                        # 点积衡量“这个 OCC 单元与这个实例消息是否匹配”。除以 sqrt(A)
                        # 控制维度增大时点积幅度，得到每条边一个标量 logits[e]。
                        logits = (query * key).sum(-1, keepdim=True) * self.attention_scale

                        # 检测分数作为注意力先验：logits + log(score) 等价于
                        # exp(logits) * score。高分实例更容易获得较大权重，但 OCC/实例
                        # 特征的匹配仍可改变实例之间的排序。clamp_min 防止 log(0)。
                        logits = logits + score_prior.clamp_min(1e-6).log()

                        # 这里只计算 softmax 的分子 exp(logits)。真正的分母在下方通过
                        # index_add 按 inverse 分组求和：同一 dst 的所有实例共用一个分母。
                        # 最终 averaged = sum(exp(logit_e)*message_e) / sum(exp(logit_e))，
                        # 即在“同一个 OCC 单元关联的实例”之间进行单头 softmax 聚合。
                        # clamp 仅限制极端 logits，避免 exp 溢出。
                        weights = logits.clamp(-20, 20).exp()
                    else:
                        # 原默认路径：仅按检测分数（及可选 edge_match）加权平均。
                        weights = score_prior
                    # inverse[e] 表示第 e 条边属于 active 中的哪个 OCC 单元。
                    # 因此 index_add 会分别计算每个 OCC 单元的加权消息和权重分母。
                    total = flat.new_zeros((len(active), channels)).index_add(0, inverse, messages * weights)
                    denom = flat.new_zeros((len(active), 1)).index_add(0, inverse, weights)
                    averaged = total / denom.clamp_min(1e-6)
                    gate = torch.sigmoid(self.gate(torch.cat([flat[active], averaged], -1)))
                    if self.aggregation == 'attention':
                        # 注意力 softmax 负责实例间分配；注意力加权置信度负责整体注入强度。
                        # 相同实例重复出现时，聚合消息和强度都保持不变。
                        confidence = flat.new_zeros((len(active), 1)).index_add(
                            0, inverse, weights * score_prior) / denom.clamp_min(1e-6)
                    else:
                        # 保持原加权平均路径的总置信度幅度控制。
                        confidence = denom.clamp(max=1)
                    update = gate * averaged * confidence
                    flat = flat.index_add(0, active, update)
                # zero 将上面的参数依赖真正接到输出/loss；两项均为0，不改变当前 BEV 数值。
                # features.sum()*0 同时保留“实例特征→来源 gather→检测特征”的反向路径：
                # 即使实例为空或均未建边，上游也可收到零梯度；空 Tensor 的 sum 同样保留计算图。
                # 这不是冻结参数：零梯度时，优化器已有动量或 AdamW 权重衰减仍可能更新参数。
                flat = flat + zero + item['features'].float().sum() * 0
                outputs.append(flat.reshape(height, width, channels).permute(2, 0, 1))
        return torch.stack(outputs).to(bev.dtype)
