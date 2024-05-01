
from collections import OrderedDict
import warnings
from torchvision.transforms.transforms import Normalize, Resize, InterpolationMode
from torch.nn import Module, Sequential, BatchNorm2d, ReLU, Flatten, Conv2d, MaxPool2d
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.ops import FrozenBatchNorm2d, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from torchvision.ops.misc import Conv2dNormActivation

from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from typing import Dict, List, Optional, Tuple

# backbone: Module
# return_layers: Dict[str, str]
# in_channels_list: List[int]
# out_channels: int
# extra_blocks: ExtraFPNBlock | None = None
# norm_layer: ((...) -> Module) | None = None


backbone_body = {
    "conv1": Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
    "bn1": FrozenBatchNorm2d(64, eps=1e-05),
    "relu": ReLU(inplace=True),
    "maxpool": MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
    "layer1": Sequential(
        Sequential(OrderedDict([       
            ("conv1", Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv2", Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv3", Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(256, eps=1e-05)),
            ("relu", ReLU(inplace=True)),
            ("downsample", Sequential(
                Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                FrozenBatchNorm2d(256, eps=1e-05)
            ))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv2", Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv3", Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(256, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict(([
            ("conv1", Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv2", Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(64, eps=1e-05)),
            ("conv3", Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(256, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])))
    ),
    "layer2": Sequential(
        Sequential(OrderedDict([ 
            ("conv1", Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv2", Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv3", Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(512, eps=1e-05)),
            ("relu", ReLU(inplace=True)),
            ("downsample", Sequential(
                Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                FrozenBatchNorm2d(512, eps=1e-05)
            ))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv2", Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv3", Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(512, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv2", Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv3", Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(512, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv2", Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(128, eps=1e-05)),
            ("conv3", Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(512, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ]))
    ),
    "layer3": Sequential(
        Sequential(OrderedDict([ 
            ("conv1", Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True)),
            ("downsample", Sequential(
                Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
                FrozenBatchNorm2d(1024, eps=1e-05)
            ))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([
            ("conv1", Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv2", Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(256, eps=1e-05)),
            ("conv3", Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(1024, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ]))
    ),
    "layer4": Sequential(
        Sequential(OrderedDict([ 
            ("conv1", Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv2", Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv3", Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(2048, eps=1e-05)),
            ("relu", ReLU(inplace=True)),
            ("downsample", Sequential(
                Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
                FrozenBatchNorm2d(2048, eps=1e-05)
            ))
        ])),
        Sequential(OrderedDict([ 
            ("conv1", Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv2", Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv3", Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(2048, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ])),
        Sequential(OrderedDict([ 
            ("conv1", Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn1", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv2", Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ("bn2", FrozenBatchNorm2d(512, eps=1e-05)),
            ("conv3", Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ("bn3", FrozenBatchNorm2d(2048, eps=1e-05)),
            ("relu", ReLU(inplace=True))
        ]))
    ),
} # type: ignore

class Backbone(nn.Module):
    def __init__(self,):
        super().__init__()

transform = GeneralizedRCNNTransform(min_size=768, max_size=768, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

class LPFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone:nn.Module, roi_heads:nn.Module, transform:nn.Module):
        self.transform = transform
        self.backbone= BackboneWithFPN(
            backbone = Backbone(),
            return_layers = {},
            in_channels_list =[0],
            out_channels = 0,
            extra_blocks = LastLevelMaxPool(),
            norm_layer = None
        )
        
        self.fpn = FeaturePyramidNetwork([0,1,2,3], out_channels=4, extra_blocks=LastLevelMaxPool())       
        #     (fpn): FeaturePyramidNetwork(
        #         (inner_blocks): ModuleList(
        #             (0): Conv2dNormActivation(
        #                 (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #             )
        #             (1): Conv2dNormActivation(
        #                 (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #             )
        #             (2): Conv2dNormActivation(
        #                 (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #             )
        #             (3): Conv2dNormActivation(
        #                 (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #             )
        #         )
        #         (layer_blocks): ModuleList(
        #             (0-3): 4 x Conv2dNormActivation(
        #                 (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #             )
        #         )
        #         (extra_blocks): LastLevelMaxPool()
        #     )
        # )

        self.rpn = RegionProposalNetwork(anchor_generator='', head=RPNHead()) # type: ignore
        # (rpn): RegionProposalNetwork(
        #     (anchor_generator): AnchorGenerator()
        #     (head): RPNHead(
        #         (conv): Sequential(
        #             (0): Conv2dNormActivation(
        #                 (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #                 (1): ReLU(inplace=True)
        #             )
        #         )
        #         (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        #         (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        #     )
        # )
        # args = "fg_iou_thresh", "bg_iou_thresh", "batch_size_per_image", "positive_fraction", "pre_nms_top_n", "post_nms_top_n", "nms_thresh"
        
        
        self.roi_heads = RoIHeads(box_roi_pool=MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)) # type: ignore
        # (roi_heads): RoIHeads(
        #     (box_roi_pool): 
        #     (box_head): TwoMLPHead(
        #         (fc6): Linear(in_features=12544, out_features=1024, bias=True)
        #         (fc7): Linear(in_features=1024, out_features=1024, bias=True)
        #     )
        #     (box_predictor): FastRCNNPredictor(
        #         (cls_score): Linear(in_features=1024, out_features=36, bias=True)
        #         (bbox_pred): Linear(in_features=1024, out_features=144, bias=True)
        #     )
        # )
        # args = "box_head", "box_predictor", "fg_iou_thresh", "bg_iou_thresh", "batch_size_per_image", "positive_fraction", "bbox_reg_weights", "score_thresh", "nms_thresh", "detections_per_img"

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting(): # type: ignore
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        