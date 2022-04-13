"""
Interface for custom data.

This module handles datasets and is the class that you need to inherit from for your custom dataset.
This class gives you all the handles so that you can train with a new –dataset=mydataset.
The particular configuration of keypoints and skeleton is specified in the headmeta instances
"""

import argparse
import torch
import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from openpifpaf.datasets import DataModule
from openpifpaf import metric, transforms
from .. import encoder, headmeta
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .constants import get_constants
from .metrics import MeanPixelError


class FurnitureKp(DataModule):
    """
    DataModule for the furniture dataset.
    """

    train_annotations = 'data-furniture/annotations_chair/pascal3d_train.json'
    val_annotations = 'data-furniture/annotations_chair/pascal3d_val.json'
    eval_annotations = val_annotations ##############################################     important
    train_image_dir = 'data-furniture/images_chair/train/'
    val_image_dir = 'data-furniture/images_chair/val/'
    eval_image_dir = val_image_dir


    n_images = None
    square_edge = 485
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()
        if self.weights is not None:
            caf_weights = []
            for bone in self.FURNITURE_SKELETON:
                caf_weights.append(max(self.weights[bone[0] - 1],
                                       self.weights[bone[1] - 1]))
            w_np = np.array(caf_weights)
            caf_weights = list(w_np / np.sum(w_np) * len(caf_weights))
        else:
            caf_weights = None
        cif = headmeta.CifFurniture('cif_furniture', 'furniture',
                           keypoints=self.FURNITURE_KEYPOINTS,
                           sigmas=self.FURNITURE_SIGMAS,
                           pose=self.FURNITURE_POSE,
                           draw_skeleton=self.FURNITURE_SKELETON,
                           score_weights=self.FURNITURE_SCORE_WEIGHTS,
                           training_weights=self.weights)
        caf = headmeta.CafFurniture('caf_furniture', 'furniture',
                           keypoints=self.FURNITURE_KEYPOINTS,
                           sigmas=self.FURNITURE_SIGMAS,
                           pose=self.FURNITURE_POSE,
                           skeleton=self.FURNITURE_SKELETON,
                           training_weights=caf_weights)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module furniture')

        group.add_argument('--furniture-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--furniture-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--furniture-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--furniture-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--furniture-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--furniture-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--furniture-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--furniture-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--furniture-no-augmentation',
                           dest='furniture_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--furniture-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--furniture-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--furniture-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--furniture-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')
        group.add_argument('--furniture-apply-local-centrality-weights',
                           dest='furniture_apply_local_centrality',
                           default=False, action='store_true',
                           help='Weigh the CIF and CAF head during training.')


        # evaluation
        assert cls.eval_annotation_filter
        group.add_argument('--furniture-no-eval-annotation-filter',
                           dest='furniture_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--furniture-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--furniture-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--furniture-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)
        group.add_argument('--furniture-use-10-kps', default=False, action='store_true',
                           help=(' '))

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # furniture specific
        cls.train_annotations = args.furniture_train_annotations
        cls.val_annotations = args.furniture_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.furniture_train_image_dir
        cls.val_image_dir = args.furniture_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.furniture_square_edge
        cls.extended_scale = args.furniture_extended_scale
        cls.orientation_invariant = args.furniture_orientation_invariant
        cls.blur = args.furniture_blur
        cls.augmentation = args.furniture_augmentation  # loaded by the dest name
        cls.rescale_images = args.furniture_rescale_images
        cls.upsample_stride = args.furniture_upsample
        cls.min_kp_anns = args.furniture_min_kp_anns
        cls.b_min = args.furniture_bmin
        if True:#args.keypoint5_use_10_kps:
            (cls.FURNITURE_KEYPOINTS, cls.FURNITURE_SKELETON, cls.HFLIP, cls.FURNITURE_SIGMAS, cls.FURNITURE_POSE,
             cls.FURNITURE_CATEGORIES, cls.FURNITURE_SCORE_WEIGHTS) = get_constants(10)
        # evaluation
        cls.eval_annotation_filter = args.furniture_eval_annotation_filter
        cls.eval_long_edge = args.furniture_eval_long_edge
        cls.eval_orientation_invariant = args.furniture_eval_orientation_invariant
        cls.eval_extended_scale = args.furniture_eval_extended_scale

        if args.furniture_apply_local_centrality:
            '''
            if args.furniture_use_10_kps:
                raise Exception("Applying local centrality weights only works with 66 kps.")
            cls.weights = training_weights_local_centrality
            '''

        else:
            cls.weights = None

    def _preprocess(self):
        encoders = (encoder.CifFurniture(self.head_metas[0], bmin=self.b_min),
                    encoder.CafFurniture(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                             1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            # transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(self.FURNITURE_KEYPOINTS, self.HFLIP), 0.5),
            rescale_t,
            transforms.RandomApply(transforms.Blur(), self.blur),
            transforms.RandomChoice(
                [transforms.RotateBy90(),
                 transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            transforms.MinSize(min_side=32.0),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])


    def train_loader(self):
        train_data = CocoLoader(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1,2,3,4],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoLoader(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1,2,3,4],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                transforms.DeterministicEqualChoice([
                    transforms.RescaleAbsolute(cls.eval_long_edge),
                    transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            ####################################################################
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    self.FURNITURE_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints, 2: self.head_metas[0].keypoints, 3: self.head_metas[0].keypoints, 4: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton, 2: self.head_metas[1].skeleton, 3: self.head_metas[1].skeleton, 4: self.head_metas[1].skeleton},
                ),
                transforms.ToCrowdAnnotations(self.FURNITURE_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
            ####################################################################
        ])

    def eval_loader(self):
        eval_data = CocoLoader(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1,2,3,4] if self.eval_annotation_filter else [1,2,3,4],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

# TODO: make sure that 24kp flag is activated when evaluating a 24kp model
    def metrics(self):
        return [metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1,2,3,4],
            iou_type='keypoints',
            keypoint_oks_sigmas=self.FURNITURE_SIGMAS
        ), MeanPixelError()]
