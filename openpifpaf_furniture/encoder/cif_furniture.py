import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescaler
from .. import headmeta
from ..visualizer import CifFurniture as CifVisualizer
from openpifpaf.utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class CifFurniture:
    meta: headmeta.CifFurniture
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return CifFurnitureGenerator(self)(image, anns, meta)


class CifFurnitureGenerator():
    def __init__(self, config: CifFurniture):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose)
        self.visualizer = config.visualizer or CifVisualizer(config.meta)

#################   Encode additional composite field for classification    ###############
        self.intensities = None
        self.class_bed = None
        self.class_chair = None
        self.class_sofa = None
        self.class_table = None
###########################################################################################
        self.fields_reg = None
        self.fields_bmin = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

#############################  Get category for classification fields encoding  #############################
        category_sets = [ann['category_id'] for ann in anns if not ann['iscrowd']]
#############################################################################################################

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.keypoints)
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets, category_sets)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
#################   Encode additional composite field for classification    ###############
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.class_bed = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.class_chair = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.class_sofa = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.class_table = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
###########################################################################################
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
#################   Encode additional composite field for classification    ###############
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan
        self.class_bed[:, p:-p, p:-p][:, bg_mask == 0] = np.nan
        self.class_chair[:, p:-p, p:-p][:, bg_mask == 0] = np.nan
        self.class_sofa[:, p:-p, p:-p][:, bg_mask == 0] = np.nan
        self.class_table[:, p:-p, p:-p][:, bg_mask == 0] = np.nan
###########################################################################################

    def fill(self, keypoint_sets, category_sets):
        for keypoints, category_id in zip(keypoint_sets, category_sets):
            self.fill_keypoints(keypoints, category_id)

    def fill_keypoints(self, keypoints, category_id):
        scale = self.rescaler.scale(keypoints)
        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            joint_scale = (
                scale
                if self.config.meta.sigmas is None
                else scale * self.config.meta.sigmas[f]
            )

            self.fill_coordinate(f, xyv, joint_scale, category_id)

    def fill_coordinate(self, f, xyv, scale, category_id):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length

#################   Encode additional composite field for classification    ###############
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return
        if minx < 0 or maxx > self.class_bed.shape[2] or \
           miny < 0 or maxy > self.class_bed.shape[1]:
            return
        if minx < 0 or maxx > self.class_chair.shape[2] or \
           miny < 0 or maxy > self.class_chair.shape[1]:
            return
        if minx < 0 or maxx > self.class_sofa.shape[2] or \
           miny < 0 or maxy > self.class_sofa.shape[1]:
            return
        if minx < 0 or maxx > self.class_table.shape[2] or \
           miny < 0 or maxy > self.class_table.shape[1]:
            return
###########################################################################################

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        mask_peak = np.logical_and(mask, sink_l < 0.7)
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

#################   Encode additional composite field for classification    ###############
        # update intensity
        if category_id == 1:
            self.class_bed[f, miny:maxy, minx:maxx][mask] = 1
            self.class_bed[f, miny:maxy, minx:maxx][mask_peak] = 1
        elif category_id == 2:
            self.class_chair[f, miny:maxy, minx:maxx][mask] = 1
            self.class_chair[f, miny:maxy, minx:maxx][mask_peak] = 1
        elif category_id == 3:
            self.class_sofa[f, miny:maxy, minx:maxx][mask] = 1
            self.class_sofa[f, miny:maxy, minx:maxx][mask_peak] = 1
        elif category_id == 4:
            self.class_table[f, miny:maxy, minx:maxx][mask] = 1
            self.class_table[f, miny:maxy, minx:maxx][mask_peak] = 1

        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        self.intensities[f, miny:maxy, minx:maxx][mask_peak] = 1.0
###########################################################################################

        # update regression
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:, mask] = sink_reg[:, mask]

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin[f, miny:maxy, minx:maxx][mask] = bmin

        # update scale
        assert np.isnan(scale) or 0.0 < scale < 100.0
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        p = self.config.padding
#################   Encode additional composite field for classification    ###############
        intensities = self.intensities[:, p:-p, p:-p]
        class_bed = self.class_bed[:, p:-p, p:-p]
        class_chair = self.class_chair[:, p:-p, p:-p]
        class_sofa = self.class_sofa[:, p:-p, p:-p]
        class_table = self.class_table[:, p:-p, p:-p]
###########################################################################################
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_bmin = self.fields_bmin[:, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

#################   Encode additional composite field for classification    ###############
        mask_valid_area(intensities, valid_area)
        mask_valid_area(class_bed, valid_area)
        mask_valid_area(class_chair, valid_area)
        mask_valid_area(class_sofa, valid_area)
        mask_valid_area(class_table, valid_area)
###########################################################################################
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
#################   Encode additional composite field for classification    ###############
            np.expand_dims(intensities, 1),
            np.expand_dims(class_bed, 1),
            np.expand_dims(class_chair, 1),
            np.expand_dims(class_sofa, 1),
            np.expand_dims(class_table, 1),
###########################################################################################
            fields_reg,
            np.expand_dims(fields_bmin, 1),
            np.expand_dims(fields_scale, 1),
        ], axis=1))
