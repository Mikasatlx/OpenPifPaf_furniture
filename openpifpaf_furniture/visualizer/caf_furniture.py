import copy
import logging

import openpifpaf
from ..annotation import Annotation
from .. import headmeta, show

try:
    import matplotlib.cm
    CMAP_BLUES_NAN = copy.copy(matplotlib.cm.get_cmap('Blues'))
    CMAP_BLUES_NAN.set_bad('white', alpha=0.5)
except ImportError:
    CMAP_BLUES_NAN = None

LOG = logging.getLogger(__name__)


class CafFurniture(openpifpaf.visualizer.base.Base):
    """Visualize CAF field."""

    def __init__(self, meta: headmeta.CafFurniture):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = show.KeypointPainter(monocolor_connections=True)
        self.annotation_painter = openpifpaf.show.AnnotationPainter(painters={'Annotation': keypoint_painter})

    def targets(self, field, *, annotation_dicts):
        assert self.meta.keypoints is not None
        assert self.meta.skeleton is not None

        annotations = [
            Annotation(
                keypoints=self.meta.keypoints,
                skeleton=self.meta.skeleton,
                sigmas=self.meta.sigmas,
            ).set(
                ann['keypoints'], fixed_score='', fixed_bbox=ann['bbox'])
            if not ann['iscrowd']
            else openpifpaf.annotation.AnnotationCrowd(['keypoints']).set(1, ann['bbox'])
            for ann in annotation_dicts
        ]
####################  Composite fields extension due to furniture classification  ###################
        self._confidences(field[:, 0])
        self._regressions(field[:, 5:7], field[:, 7:9], field[:, 11], field[:, 12],
                          annotations=annotations)
#####################################################################################################

####################  Composite fields extension due to furniture classification  ###################
    def predicted(self, field):
        self._confidences(field[:, 1])
        self._classifications_bed(field[:, 2])
        self._classifications_chair(field[:, 3])
        self._classifications_sofa(field[:, 4])
        self._classifications_table(field[:, 5])
        self._regressions(field[:, 6:8], field[:, 8:10], field[:, 10], field[:, 11],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 1],
                          uv_is_offset=False)
#####################################################################################################
    def _confidences(self, confidences):
        for f in self.indices('confidence'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

####################  Composite fields extension due to furniture classification  ###################
    def _classifications_bed(self, classifications_bed):
        for f in self.indices('class_bed'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(classifications_bed[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _classifications_chair(self, classifications_chair):
        for f in self.indices('class_bed'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(classifications_chair[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _classifications_sofa(self, classifications_sofa):
        for f in self.indices('class_sofa'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(classifications_sofa[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _classifications_table(self, classifications_table):
        for f in self.indices('class_table'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(classifications_table[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)
#####################################################################################################

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        for f in self.indices('regression'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                openpifpaf.show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax, annotations, color='lightgray')
                q1 = openpifpaf.show.quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                 cmap='Blues', clim=(0.5, 1.0), width=0.001)
                openpifpaf.show.quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001)
                openpifpaf.show.boxes(ax, scale_fields1[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields1[f, :2],
                           xy_scale=self.meta.stride, cmap='Blues', fill=False,
                           regression_field_is_offset=uv_is_offset)
                openpifpaf.show.boxes(ax, scale_fields2[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields2[f, :2],
                           xy_scale=self.meta.stride, cmap='Greens', fill=False,
                           regression_field_is_offset=uv_is_offset)

                self.colorbar(ax, q1)
