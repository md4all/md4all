import torch
import pytorch_lightning as pl

from models.md2.resnet_encoder import ResnetEncoder
from models.md2.pose_decoder import PoseDecoder
from data.transforms import NormalizeDynamic


class PoseNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        encoder_meta = cfg.MODEL.POSE.ENCODER.VERSION.split('-')
        assert encoder_meta[0].lower() in ['resnet']

        self.encoder = ResnetEncoder(num_layers=int(encoder_meta[1]),
                                     pretrained=cfg.MODEL.POSE.ENCODER.PRETRAINED,
                                     num_input_images=cfg.MODEL.POSE.ENCODER.NUM_INPUT_IMAGES)
        self.decoder = PoseDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                   num_input_features=cfg.MODEL.POSE.DECODER.NUM_INPUT_FEATURES,
                                   num_frames_to_predict_for=cfg.MODEL.POSE.DECODER.NUM_FRAMES_TO_PREDICT_FOR)

        self.temporal_ordering = cfg.MODEL.POSE.TEMPORAL_ORDERING

        self.normalize = NormalizeDynamic(cfg)

    def forward(self, key_img, ref_images, daytime):
        poses = []
        key_image = self.normalize(key_img, daytime)
        for i, ref_img in enumerate(ref_images):
            ref_image = self.normalize(ref_img, daytime)
            features = self.encoder(torch.cat([key_image, ref_image], dim=1)) if not self.temporal_ordering or i > 0 else self.encoder(torch.cat([ref_image, key_image], dim=1))
            axisangle, translation = self.decoder([features])
            poses.append(torch.cat([translation[:, 0], axisangle[:, 0]], dim=2))
        return torch.cat(poses, dim=1)
