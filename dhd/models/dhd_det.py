# @Author : lqf
# @Time : 2025/4/30 22:18
# Modified for 3D object detection task
# ------------------------------------------------------------------------
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhd.models.encoder import Encoder
from dhd.utils.geometry import (VoxelsSumming,
                                calculate_birds_eye_view_parameters,
                                cumulative_warp_features)
from dhd.utils.network import (pack_sequence_dim, set_bn_momentum,
                               unpack_sequence_dim)
import numpy as np
import torchvision.ops as ops
from dhd.models.fusion_model import *

class DistanceAwareDeformConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DistanceAwareDeformConv2D, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=1, box_parameters=7):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes  # Number of object classes
        self.num_anchors = num_anchors  # Anchors per grid cell
        self.box_parameters = box_parameters  # (x, y, z, l, w, h, yaw)

        # Classification head: objectness + class probabilities
        self.cls_head = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=1)
        # Regression head: box parameters
        self.reg_head = nn.Conv2d(in_channels, num_anchors * box_parameters, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H_bev, W_bev)
        cls_preds = self.cls_head(x)  # (B, num_anchors * (num_classes + 1), H_bev, W_bev)
        reg_preds = self.reg_head(x)  # (B, num_anchors * box_parameters, H_bev, W_bev)

        # Reshape outputs
        B, _, H, W = cls_preds.shape
        cls_preds = cls_preds.view(B, self.num_anchors, self.num_classes + 1, H, W)
        reg_preds = reg_preds.view(B, self.num_anchors, self.box_parameters, H, W)

        return cls_preds, reg_preds

class DHD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # BEV parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        # Fusion model (SISW for collaboration)
        self.fusion = SISW() if self.cfg.MODEL.NAME == 'SISW' else None

        # Detection head
        self.detection_head = DetectionHead(
            in_channels=self.encoder_out_channels,
            num_classes=len(self.cfg.DETECTION.CLASSES),
            num_anchors=self.cfg.DETECTION.NUM_ANCHORS,
            box_parameters=7  # (x, y, z, l, w, h, yaw)
        )

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def custom_linspace(self, start, end, num_points):
        linear_points = np.linspace(0, 1, num_points)
        transformed_points = np.cbrt(linear_points)
        scaled_points = end - (end - start) * transformed_points
        scaled_points = sorted(scaled_points)
        scaled_points = torch.tensor(scaled_points, dtype=torch.float)
        return scaled_points

    def create_height_frustum(self):
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample
        start, end, step = self.cfg.LIFT.D_BOUND
        height_grid = self.custom_linspace(start, end, int((end - start) / step))
        height_grid = height_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = height_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        frustum = torch.stack((x_grid, y_grid, height_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def gbg(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        BATCH, N, _ = translation.shape
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        R = rotation
        T = translation.unsqueeze(3)
        K = intrinsics
        K_cpu = K.to('cpu')
        A = (R.cpu().matmul(torch.inverse(K_cpu))).cuda()
        B = T
        H = -50
        points = points.repeat(BATCH, N, 1, 1, 1, 1, 1)
        a21 = A[:, :, 2, 0].view(BATCH, N, 1, 1, 1, 1)
        a22 = A[:, :, 2, 1].view(BATCH, N, 1, 1, 1, 1)
        a23 = A[:, :, 2, 2].view(BATCH, N, 1, 1, 1, 1)
        b2 = B[:, :, 2, 0].view(BATCH, N, 1, 1, 1, 1)
        x = points[:, :, :, :, :, 0]
        y = points[:, :, :, :, :, 1]
        z = points[:, :, :, :, :, 2]
        xiebian = (H - b2) / (a21 * x + a22 * y + a23)
        short_xiebian = z * xiebian / -H
        new_z = xiebian - short_xiebian
        points[:, :, :, :, :, 2] = new_z
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        rotation_cpu = rotation.cpu()
        intrinsics_cpu = intrinsics.cpu()
        result_cpu = rotation_cpu.matmul(torch.inverse(intrinsics_cpu))
        combined_transformation = result_cpu.cuda()
        points = combined_transformation.view(BATCH, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(BATCH, N, 1, 1, 1, 3)
        return points

    def encoder_forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def projection_to_birds_eye_view(self, x, geometry):
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )
        N = n * d * h * w
        for b in range(batch):
            x_b = x[b].reshape(N, c)
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)
            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)
            output[b] = bev_feature
        return output

    def generate_each_bev(self, features, geometry):
        bev_results = []
        for i in range(features.size(1)):
            x_slice = features[:, i:i+1, ...]
            geometry_slice = geometry[:, i:i+1, ...]
            bev_result = self.projection_to_birds_eye_view(x_slice, geometry_slice)
            bev_results.append(bev_result)
        bev_tensor = torch.stack(bev_results, dim=1)
        return bev_tensor

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, is_train):
        b, s, n, c, h, w = x.shape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)
        geometry = self.gbg(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        if self.fusion is None:
            x = self.projection_to_birds_eye_view(x, geometry)
            bandwidth = 1
        else:
            bev_features = self.generate_each_bev(features=x, geometry=geometry)
            res = self.fusion(bev_features, is_train)
            if len(res) == 2:
                x = res[0]
                bandwidth = res[1]
            else:
                x = res
                bandwidth = 1
        x = unpack_sequence_dim(x, b, s)
        return x, bandwidth

    def forward(self, image, intrinsics, extrinsics, future_egomotion, is_train=True):
        output = {}
        start_time = time.time()

        # Process only the current frame (no temporal sequence needed for detection)
        image = image[:, 0:1].contiguous()
        intrinsics = intrinsics[:, 0:1].contiguous()
        extrinsics = extrinsics[:, 0:1].contiguous()
        future_egomotion = future_egomotion[:, 0:1].contiguous()

        # Calculate BEV features
        x, bandwidth = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, is_train)

        # Detection head
        cls_preds, reg_preds = self.detection_head(x[:, 0])  # Process current frame

        perception_time = time.time()

        output['cls_preds'] = cls_preds
        output['reg_preds'] = reg_preds
        output['band_width'] = bandwidth
        output['perception_time'] = perception_time - start_time
        output['total_time'] = output['perception_time']

        return output

    def compute_loss(self, output, targets):
        cls_preds = output['cls_preds']  # (B, num_anchors, num_classes + 1, H, W)
        reg_preds = output['reg_preds']  # (B, num_anchors, 7, H, W)
        B, num_anchors, _, H, W = cls_preds.shape

        # Reshape predictions
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2).contiguous().view(B, H * W * num_anchors, -1)
        reg_preds = reg_preds.permute(0, 3, 4, 1, 2).contiguous().view(B, H * W * num_anchors, 7)

        # Generate anchor grid
        x = torch.linspace(self.cfg.LIFT.X_BOUND[0], self.cfg.LIFT.X_BOUND[1], self.bev_dimension[0], device=cls_preds.device)
        y = torch.linspace(self.cfg.LIFT.Y_BOUND[0], self.cfg.LIFT.Y_BOUND[1], self.bev_dimension[1], device=cls_preds.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        anchors = torch.stack([x, y], dim=-1).view(-1, 2)  # (H*W, 2)
        anchors = anchors.repeat(num_anchors, 1)  # (H*W*num_anchors, 2)

        cls_loss = 0
        reg_loss = 0
        num_pos = 0

        for b in range(B):
            # Ground truth: list of boxes with [x, y, z, l, w, h, yaw, class_id]
            gt_boxes = targets[b]['boxes_3d']  # Shape: (N, 8) [x, y, z, l, w, h, yaw, class_id]
            if len(gt_boxes) == 0:
                cls_target = torch.zeros_like(cls_preds[b])
                cls_loss += F.binary_cross_entropy_with_logits(cls_preds[b], cls_target)
                continue

            # Assign anchors to ground truth
            gt_centers = gt_boxes[:, :2]  # (N, 2)
            distances = torch.cdist(anchors, gt_centers)  # (H*W*num_anchors, N)
            min_distances, min_indices = distances.min(dim=1)
            pos_mask = min_distances < self.cfg.DETECTION.POS_THRESHOLD
            num_pos += pos_mask.sum()

            # Classification targets
            cls_target = torch.zeros_like(cls_preds[b])
            pos_indices = torch.where(pos_mask)[0]
            for idx in pos_indices:
                gt_idx = min_indices[idx]
                class_id = int(gt_boxes[gt_idx, 7])
                cls_target[idx, class_id + 1] = 1  # +1 for background class

            # Regression targets
            reg_target = torch.zeros_like(reg_preds[b])
            for idx in pos_indices:
                gt_idx = min_indices[idx]
                gt_box = gt_boxes[gt_idx, :7]  # [x, y, z, l, w, h, yaw]
                anchor_center = anchors[idx]
                reg_target[idx] = torch.tensor([
                    gt_box[0] - anchor_center[0],  # dx
                    gt_box[1] - anchor_center[1],  # dy
                    gt_box[2],                    # z
                    gt_box[3],                    # l
                    gt_box[4],                    # w
                    gt_box[5],                    # h
                    gt_box[6]                     # yaw
                ], device=reg_preds.device)

            # Losses
            cls_loss += F.binary_cross_entropy_with_logits(cls_preds[b], cls_target)
            if pos_mask.sum() > 0:
                reg_loss += F.smooth_l1_loss(reg_preds[b][pos_mask], reg_target[pos_mask])

        cls_loss = cls_loss / B
        reg_loss = reg_loss / max(num_pos, 1)
        loss = self.cfg.DETECTION.LOSS_WEIGHTS.CLS * cls_loss + self.cfg.DETECTION.LOSS_WEIGHTS.REG * reg_loss

        return loss, {'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item(), 'num_pos': num_pos.item()}