import pickle
from scipy import io
import numpy as np
import matplotlib.pyplot as plt


def load_lidar_data(fn):
    d = io.loadmat(fn)
    r = []
    for m in d['lidar'][0]:
        t = {}
        t['t'] = m[0][0][0][0][0]  # Unix time in seconds
        nn = len(m[0][0])
        if (nn != 5) and (nn != 6):
            raise ValueError('Corrupted lidar data')

        # +x axis points forwards, +y points left, +z points upwards
        t['xyth'] = m[0][0][nn - 4][0]  # xy yaw of lidar
        t['resolution'] = m[0][0][nn - 3][0][0]  # resolution in radians
        t['rpy'] = m[0][0][nn - 2][0]  # roll-pitch-yaw
        t['scan'] = m[0][0][nn - 1][0]
        r.append(t)

    # we are going to remove the initial yaw from the lidar yaw here
    offset = r[0]['rpy'][2]
    for i in range(len(r)):
        r[i]['rpy'][2] -= offset
    return r


def load_joint_data(fn):
    keys = ['acc', 'ts', 'rpy', 'gyro', 'pos', 'ft_l', 'ft_r', 'head_angles']
    d = io.loadmat(fn)
    j = {k: d[k] for k in keys}
    j['t'] = j['ts'].squeeze()
    j['xyz'] = j['pos']
    j.pop('ts')
    j.pop('pos')
    return j


# these are some functions to visualize lidar data
def show_lidar(d):
    # angles of each lidar ray are in a field of view [-135, 135] degree about
    # the optical axis
    th = np.arange(0, 270.25, 0.25) * np.pi / 180.0

    plt.figure(1)
    plt.clf()
    ax = plt.subplot(111, projection='polar')
    try:
        for i in range(200, len(d), 10):
            d[i]['scan'][d[i]['scan'] > 30] = 30

            ax.clear()
            ax.plot(th, d[i]['scan'])
            ax.plot(th, d[i]['scan'], 'r.')
            ax.set_rmax(10)
            ax.set_rticks([0.5, 1, 1.5, 2])
            ax.set_rlabel_position(-22.5)
            ax.grid(True)
            ax.set_title('Lidar scans [%d]: %2.3f [sec]' % (i, d[i]['t']))
            plt.draw()
            plt.pause(1e-3)

    except KeyboardInterrupt:
        plt.close(1)


joint_names = ['Neck', 'Head', 'ShoulderL', 'ArmUpperL', 'LeftShoulderYaw', 'ArmLowerL', 'LeftWristYaw',
               'LeftWristRoll', 'LeftWristYaw2', 'PelvYL', 'PelvL', 'LegUpperL', 'LegLowerL', 'AnkleL', 'FootL',
               'PelvYR', 'PelvR', 'LegUpperR', 'LegLowerR', 'AnkleR', 'FootR', 'ShoulderR', 'ArmUpperR',
               'RightShoulderYaw', 'ArmLowerR', 'RightWristYaw', 'RightWristRoll', 'RightWristYaw2', 'TorsoPitch',
               'TorsoYaw', 'l_wrist_grip1', 'l_wrist_grip2', 'l_wrist_grip3', 'r_wrist_grip1', 'r_wrist_grip2',
               'r_wrist_grip3', 'ChestLidarPan']
joint_name_to_index = {k: v for v, k in zip(range(len(joint_names)), joint_names)}
joint_index_to_name = {v: k for v, k in zip(range(len(joint_names)), joint_names)}


if __name__ == '__main__':

    import os

    # load the data
    idx = 0
    split = 'train'
    lidar = load_lidar_data(os.path.join('data/%s/%s_lidar%d' % (split, split, idx)))
    joint = load_joint_data(os.path.join('data/%s/%s_joint%d' % (split, split, idx)))

    print('lidar:', lidar[0].keys())  # lidar: dict_keys(['t', 'xyth', 'resolution', 'rpy', 'scan'])

    print('joint:', joint.keys())  # joint: dict_keys(['acc', 'rpy', 'gyro', 'ft_l', 'ft_r', 'head_angles', 't', 'xyz'])

    # visualize the lidar data
    show_lidar(lidar)

    print('done')
