
import os
import time
import numpy as np
import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motGPT.render.pyrender.smpl_render import SMPLRender

SMPL_MODEL_PATH = 'deps/smpl_models/smpl'

def render_motion(data, feats, output_dir, fname=None, method='slow', smpl_model_path=SMPL_MODEL_PATH, fps=20, text=None):    
    if fname is None:
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
            time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # np.save(output_npy_path, feats)

    # SEM if / elif

    if len(data.shape) == 4:
        data = data[0]

    data = data - data[0, 0]

    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(data)

    pose = np.concatenate([
        pose,
        np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
    ], 1)

    shape = [768, 768]
    render = SMPLRender(smpl_model_path)

    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])

    vid = []

    aroot = data[:, 0].copy()
    aroot[:, 1] = -aroot[:, 1]
    aroot[:, 2] = -aroot[:, 2]

    params = dict(
        pred_shape=np.zeros([1, 10]),
        pred_root=aroot,
        pred_pose=pose
    )

    render.init_renderer([shape[0], shape[1], 3], params)

    for i in range(data.shape[0]):
        renderImg = render.render(i)
        vid.append(renderImg)

    out_video = mp.ImageSequenceClip(vid, fps=fps)
    out_video.write_videofile(output_mp4_path, fps=fps)

    del render