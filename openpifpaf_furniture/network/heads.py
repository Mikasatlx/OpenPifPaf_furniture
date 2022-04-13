"""Head networks."""

import argparse
import functools
import logging
import math

import torch

import openpifpaf
from .. import headmeta

LOG = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, unsqueeze=(0, 0)):
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)

    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)

    return xy


class CompositeFieldFurniture(openpifpaf.network.heads.HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True

    def __init__(self,
                 meta: openpifpaf.headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_class, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
####################  Composite fields extension due to furniture classification  ###################
        self.n_components = 1 + meta.n_confidences + meta.n_class + meta.n_vectors * 2 + meta.n_scales
#####################################################################################################
        self.conv = torch.nn.Conv2d(
            in_features, meta.n_fields * self.n_components * (meta.upsample_stride ** 2),
            kernel_size, padding=padding, dilation=dilation,
        )

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeFieldFurniture')
        group.add_argument('--new-cf4-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--new-cf4-no-inplace-ops', dest='cf4_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.cf4_dropout
        cls.inplace_ops = args.cf4_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.n_components,
            feature_height,
            feature_width
        )
        
        if not self.training and self.inplace_ops:
            # classification
            intensity_x = x[:, :, 1:1 + self.meta.n_confidences]
            torch.sigmoid_(intensity_x)
            #print('intensity')
            #print(intensity_x[0,1,:,46,43])
            
            class_x = x[:, :, 1 + self.meta.n_confidences:1 + self.meta.n_confidences + self.meta.n_class]
            class_x = torch.softmax(class_x, dim=2)
            x[:, :, 1 + self.meta.n_confidences:1 + self.meta.n_confidences + self.meta.n_class] = class_x
            #print('softmax')
            #print(class_x[0,1,:,46,43])

####################  Composite fields extension due to furniture classification  ###################
            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = 1 + self.meta.n_confidences + self.meta.n_class
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_class + self.meta.n_vectors * 2
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
#####################################################################################################
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # width
            width_x = x[:, 0:1]

            # classification
            intensity_x = x[:, 1:1 + self.meta.n_confidences]
            intensity_x = torch.sigmoid(intensity_x)

            # regressions x
            first_reg_feature = 1 + self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            # regressions x: add index
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            # TODO: coreml export does not work with the index_field creation in the graph.
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [reg_x + index_field if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat
            x = torch.cat([width_x, intensity_x, *regs_x, scales_x], dim=1)

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)
        
        return x
