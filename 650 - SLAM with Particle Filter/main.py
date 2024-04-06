import click, tqdm, random
import matplotlib.pyplot as plt
from slam import *


def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create two figures after you run it.
    The first one is the robot location trajectory using odometry information obtained form the lidar.
    The second is the trajectory using the PF with a very small dynamics noise.
    The two figures should look similar.
    """
    slam = slam_t(Q=1e-8 * np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []  # x, y, theta (yaw) of the lidar
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1], p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1)
    plt.clf()
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:, 0], xyth[:, 1])
    logging.info('> Saving odometry plot in ' + os.path.join(log_dir, 'odometry_%s_%02d.jpg' % (split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg' % (split, idx)))

    # dynamics propagation using particle filter
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar for checking in this function
    n = 3  # n: number of particles
    w = np.ones(n) / float(n)  # w: weights
    p = np.zeros((3, n), dtype=np.float64)  # p: particles (3 dimensions, n particles)
    slam.init_particles(n, p, w)
    slam.p[:, 0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)  # maintains all particles across all time steps
    plt.figure(2)
    plt.clf()
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0 + 1, T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d' % t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    plt.axis('equal')
    logging.info('> Saving plot in ' + os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg' % (split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg' % (split, idx)), dpi=300)


def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step.
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]]).

    * Note that the particle array has the shape 3 x num_particles so the first particle is at [x=0.2, y=0.4, z=0.1]

    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle
    is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']  # extract the xyth from the lidar data
    xyth[2] = slam.lidar[t0]['rpy'][2]  # extract the yaw from the lidar data
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1, p=xyth.reshape((3, 1)), w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n) / float(n)
    # p = np.array([[0.2, 2, 3], [0.4, 2, 5], [0.1, 2.7, 4]])
    p = np.array([[2, 0.2, 3], [2, 0.4, 5], [2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))


def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    noise_xy = 2e-4
    noise_z = 1e-4
    slam = slam_t(resolution=0.05, Q=np.diag([noise_xy, noise_xy, noise_z]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)

    # Again initialize the map to enable calculation of the observation logp in future steps.
    # This time we want to be more careful and initialize with the correct lidar scan.
    # First find the time t0 around which we have both LiDAR data and joint data.
    #### DONE: XXXXXXXXXXX

    # I noticed that in the test data, the 2048th joint data has the closest timestamp to the 0th lidar data
    # so I will use the 2048th joint data to initialize the map
    # The interval of the joint data is about 0.0084 seconds, and the lidar about 0.024880 seconds
    joint_timestamps = slam.joint['t']
    lidar_timestamp = slam.lidar[0]['t']  # First LiDAR timestamp
    t0 = slam.lidar[0]['t']  # Initialize t0 to the first LiDAR timestamp

    # Initialize the occupancy grid using one particle and calling the observation_step function
    #### DONE: XXXXXXXXXXX

    # Assuming t0 has been correctly identified:
    initial_xyth = slam.lidar[0]['xyth']  # Extract the xyth from the lidar data
    initial_xyth[2] = slam.lidar[0]['rpy'][2]  # Extract the yaw from the lidar data

    # Initialize with one particle at lidar position
    slam.init_particles(n=1, p=initial_xyth.reshape((3, 1)), w=np.array([1]).astype(float))
    slam.observation_step(0)

    # SLAM, save data to be plotted later.
    #### DONE: XXXXXXXXXXX

    n_particles = 100  # Number of particles
    estimated_pose = []  # Estimated pose list
    # Initialize particles around the first pose with some small random variation
    initial_xyth_expanded = np.tile(initial_xyth, (n_particles, 1)).T
    noise = np.random.randn(3, n_particles) * np.array([0.1, 0.1, 0.05])[:, None]
    p = initial_xyth_expanded + noise  # Small noise around initial pose
    w = np.ones(n_particles) / n_particles  # Equal initial weight
    slam.init_particles(n=n_particles, p=p.reshape((3, n_particles)), w=w)
    slam.dynamics_step(0)

    for t in tqdm.tqdm(range(1, T)):  # Start at 1 since we've already initialized at t0, use tqdm for progress bar
        slam.dynamics_step(t)
        slam.observation_step(t)
        estimated_pose.append(slam.estimated_pose)
        # Visualize the map and particles at certain steps
        # if t % 50 == 0:
        #     print('Visualizing map at t =', t)

    estimated_pose = np.array(estimated_pose)  # Convert to numpy array
    fig = plt.figure()
    bound = np.where(slam.map.cells > 0.5)  # Occupied cells
    pose_x, pose_y = slam.map.grid_cell_from_xy(estimated_pose[:, 0], estimated_pose[:, 1])  # Convert pose to grid cell
    plt.plot(pose_x, pose_y, 'r', markersize=0.5)  # Plot estimated pose
    plt.plot(bound[0], bound[1], 'k.', markersize=0.5)  # Plot occupied cells
    plt.xlim([0, slam.map.szx])
    plt.ylim([0, slam.map.szy])
    # plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('SLAM Map')
    plt.savefig(os.path.join(log_dir, 'slam_map_%s_%02d.jpg' % (split, idx)), dpi=500)  # Save map plot
    plt.close(fig)  # Close figure


@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s' % mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        run_slam(src_dir, log_dir, idx, split)


if __name__ == '__main__':
    # If not testing
    main()

    # # If testing
    # import cProfile
    # cProfile.run('main()', 'main.prof')
    # import pstats
    # p = pstats.Stats('main.prof')
    # p.sort_stats('cumulative').print_stats(10)


