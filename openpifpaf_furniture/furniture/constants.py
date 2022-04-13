import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf

FURNITURE_KEYPOINTS_10 = [
    "leg_front_left",   # 1
    "leg_front_right",  # 2
    "leg_back_left",    # 3
    "leg_back_right",   # 4
    "seat_front_left",  # 5
    "seat_front_right", # 6
    "seat_back_left",   # 7
    "seat_back_right",  # 8
    "top_left",         # 9
    "top_right"         # 10
]

FURNITURE_SKELETON_10 = [
    [1, 5], 
    [2, 6],
    [3, 7],
    [4, 8],
    [5, 6],
    [5, 7],
    [5, 9],
    [6, 8],
    [6, 10],
    [7, 8],
    [7, 9],
    [8, 10],
    [9, 10]
]

FURNITURE_SIGMAS_10 = [0.05] * 10

FURNITURE_SCORE_WEIGHTS_10 = [3.0] * 10

HFLIP_10 = {
    'leg_front_left':'leg_front_right',
    'leg_front_right':'leg_front_left',
    'leg_back_left':'leg_back_right',
    'leg_back_right':'leg_back_left',
    'seat_front_left':'seat_front_right',
    'seat_front_right':'seat_front_left',
    'seat_back_left':'seat_back_right',
    'seat_back_right':'seat_back_left',
    'top_left':'top_right',
    'top_right':'top_left'
}

FURNITURE_CATEGORIES_10 = [1,2,3,4]

FURNITURE_POSE_10 = np.array([
    [-4, -2, 2],         #leg_front_left
    [2.04, -2.37, 2],    #leg_front_right
    [-1.59, -0.03, 2],   #leg_back_left
    [3.24, -0.17, 2],    #leg_back_right
    [-4.05, 1.74, 2],    #seat_front_left
    [1.98, 1.46, 2],     #seat_front_right
    [-1.62, 3.92, 2],    #seat_back_left
    [3.26, 3.8, 2],      #seat_back_right
    [-1.59, 7.46, 2],    #top_left
    [3.24, 7.43, 2]      #top_right
])

FURNITURE_POSE_FRONT_10 = np.array([
    [-4, -2, 2],         #leg_front_left
    [4, -2, 2],    #leg_front_right
    [-2.65, -1.6, 2],   #leg_back_left
    [2.75, -1.65, 2],    #leg_back_right
    [-4.02, 2.6, 2],    #seat_front_left
    [4.04, 2.54, 2],     #seat_front_right
    [-2.65, 2.2, 2],    #seat_back_left
    [2.61, 2.12, 2],      #seat_back_right
    [-2.65, 7.17, 2],    #top_left
    [2.61, 7.2, 2]      #top_right
])

FURNITURE_POSE_REAR_10 = np.array([
    [-4, -2, 2],         #leg_front_left
    [4, -2, 2],    #leg_front_right
    [-2.65, -1.6, 2],   #leg_back_left
    [2.75, -1.65, 2],    #leg_back_right
    [-4.02, 2.6, 2],    #seat_front_left
    [4.04, 2.54, 2],     #seat_front_right
    [-2.65, 2.2, 2],    #seat_back_left
    [2.61, 2.12, 2],      #seat_back_right
    [-4.05, 7.23, 2],    #top_left
    [4.01, 7.23, 2]      #top_right
])

FURNITURE_POSE_LEFT_10 = np.array([
    [-4, -2, 2],         #leg_front_left
    [4, -2, 2],    #leg_front_right
    [-2.65, -1.6, 2],   #leg_back_left
    [2.75, -1.65, 2],    #leg_back_right
    [-4.02, 2.6, 2],    #seat_front_left
    [4.04, 2.54, 2],     #seat_front_right
    [-2.65, 2.2, 2],    #seat_back_left
    [2.61, 2.12, 2],      #seat_back_right
    [-4.05, 7.23, 2],    #top_left
    [-2.62, 6.34, 2]      #top_right
])

FURNITURE_POSE_RIGHT_10 = np.array([
    [-4, -2, 2],         #leg_front_left
    [4, -2, 2],    #leg_front_right
    [-2.65, -1.6, 2],   #leg_back_left
    [2.75, -1.65, 2],    #leg_back_right
    [-4.02, 2.6, 2],    #seat_front_left
    [4.04, 2.54, 2],     #seat_front_right
    [-2.65, 2.2, 2],    #seat_back_left
    [2.61, 2.12, 2],      #seat_back_right
    [2.64, 5.6, 2],    #top_left
    [4.04, 7.23, 2]      #top_right
])

def get_constants(num_kps):
    if num_kps == 10:
        FURNITURE_POSE_10[:, 2] = 2
        return [FURNITURE_KEYPOINTS_10, FURNITURE_SKELETON_10, HFLIP_10, FURNITURE_SIGMAS_10,
                FURNITURE_POSE_10, FURNITURE_CATEGORIES_10, FURNITURE_SCORE_WEIGHTS_10]
    # using no if-elif-else construction due to pylint no-else-return error
    raise Exception("Only the pose with 10 keypoints is available.")

def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)

def print_associations():
    print("\nAssociations of the furniture skeleton with 10 keypoints")
    for j1, j2 in FURNITURE_SKELETON_10:
        print(FURNITURE_KEYPOINTS_10[j1 - 1], '-', FURNITURE_KEYPOINTS_10[j2 - 1])

def main():
    print_associations()
# =============================================================================
#     draw_skeletons(FURNITURE_POSE_10, sigmas = FURNITURE_SIGMAS_10, skel = FURNITURE_SKELETON_10,
#                    kps = FURNITURE_KEYPOINTS_10, scr_weights = FURNITURE_SCORE_WEIGHTS_10)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, FURNITURE_POSE_10, FURNITURE_SKELETON_10)
        anim_24.save('openpifpaf/plugins/furniture/docs/FURNITURE_10_Pose.gif', fps=30)


if __name__ == '__main__':
    main()