# register ops first
from . import cpp_extension
cpp_extension.register_ops()

import openpifpaf
from . import decoder
from . import encoder
from . import network
from . import furniture
from . import show
from . import visualizer
from . import annotation
from . import headmeta

# load plugins last
def register():
    openpifpaf.DATAMODULES['furniture'] = furniture.furniture_kp.FurnitureKp
    openpifpaf.HEADS[headmeta.CifFurniture] = network.heads.CompositeFieldFurniture
    openpifpaf.HEADS[headmeta.CafFurniture] = network.heads.CompositeFieldFurniture
    openpifpaf.LOSSES[headmeta.CifFurniture] = network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.CafFurniture] = network.losses.CompositeLoss
    openpifpaf.decoder.DECODERS.add(decoder.CifCafFurniture)
    openpifpaf.show.annotation_painter.PAINTERS["Annotation"] = show.painters.KeypointPainter
