import numpy as np
from scipy import io
from quaternion import Quaternion
import math

from matplotlib import pyplot as plt

# data files are numbered on the server.
# for example imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an unscented kalman filter

def compute_imu_data(raw_data, alpha, beta):
    # raw_data is a 1xN matrix where N is the number of samples
    # alpha and beta are the calibration parameters
    # returns a 1xN matrix of calibrated data
    return (raw_data - beta) * 3300.0 / 1023.0 / alpha


def solve_scale_factors(omega, v_prime):
    # Make sure the length of omega and v_prime are the same
    # If not the same, use the shorter one
    if len(omega) > len(v_prime):
        omega = omega[0:len(v_prime)]
    if len(v_prime) > len(omega):
        v_prime = v_prime[0:len(omega)]
    S = np.sum(omega * v_prime) / np.sum(omega ** 2)
    return S


def sigma_points_gen_and_trans(x_km1, P_km1, Q, dt, i):
    Sigma_X_i = np.zeros((7, 12))
    Sigma_Y_i = np.zeros((7, 12))

    S = np.linalg.cholesky(np.sqrt(6) * (P_km1 + Q))  # Eq (35)
    # W_{i,i+n} = columns of (±sqrt(2n)S)
    W_maker = np.hstack((S, -S))  # To compute set {W_i}
    q_x = Quaternion(x_km1[0, 0], x_km1[1:4, 0])
    q_x.normalize()
    for j in range(12):
        # Generate sigma points Sigma_X_i
        # q_W is the quaternion to the first three components of W_i
        q_W = Quaternion()
        q_W.from_axis_angle(W_maker[0:3, j])
        q_X = q_x * q_W  # Eq (34)
        q_X.normalize()
        Sigma_X_i[0:4, j] = q_X.q
        Sigma_X_i[4:7, j] = x_km1[4:7, 0] + W_maker[3:6, j]

        # Transform sigma points
        q_d = Quaternion()
        q_d.from_axis_angle(Sigma_X_i[4:7, j] * dt[i])
        q_Y = q_X * q_d
        q_Y.normalize()
        Sigma_Y_i[0:4, j] = q_Y.q
        Sigma_Y_i[4:7, j] = Sigma_X_i[4:7, j]

    return Sigma_X_i, Sigma_Y_i


def compute_mean_quaternion(q_Sigma_Y_i, q_init):
    # Define GD parameters
    prev_e_norm = 5.0
    thres = 1e-4

    e_hat = np.zeros(3)
    q_mean = Quaternion(q_init[0], q_init[1:4])
    count = 0
    while np.abs(np.linalg.norm(e_hat) - prev_e_norm) > thres and count < 70:
        prev_e_norm = np.linalg.norm(e_hat)
        e_i = np.zeros((3, 12))
        for i in range(12):
            q_Sigma_Y_i_i = Quaternion(q_Sigma_Y_i[0, i], q_Sigma_Y_i[1:4, i])
            q_W = q_Sigma_Y_i_i * q_mean.inv()
            q_W.normalize()
            e_i[:, i] = q_W.axis_angle()
        e_hat = np.mean(e_i, axis=1)
        q_e = Quaternion()
        q_e.from_axis_angle(e_hat)
        q_mean = q_e * q_mean
        q_mean.normalize()
        count += 1

    return q_mean


def compute_mean_Y(quat_angle_7, q_init):
    # Eq (38)
    result = np.zeros((7, 1))
    # First we need to compute the mean of quaternions, Sigma_Y_i[0:4, :] and Sigma_X_i[0:4, 0]
    q_mean = compute_mean_quaternion(quat_angle_7[0:4, :], q_init)
    result[0:4, 0] = q_mean.q
    result[4:7, 0] = np.mean(quat_angle_7[4:7, :], axis=1)
    return result


def compute_W_residual(Sigma_Y_i, Y_mean):
    W_residual = np.zeros((6, 12))
    q_mean = Quaternion(Y_mean[0,0], Y_mean[1:4,0])
    omega_mean = Y_mean[4:7, 0]

    for i in range(12):
        q_Sigma_Y_i_i = Quaternion(Sigma_Y_i[0, i], Sigma_Y_i[1:4, i])
        r_W = q_Sigma_Y_i_i * q_mean.inv()
        r_W.normalize()
        r_W = r_W.axis_angle()
        omega_W = Sigma_Y_i[4:7, i] - omega_mean
        W_residual[:, i] = np.hstack((r_W, omega_W))

    return W_residual


def compute_Y_cov(W_residual):
    Y_cov = (W_residual @ W_residual.T) / 12.0
    return Y_cov


def compute_Z(Sigma_Y_i):
    Z = np.zeros((6, 12))
    q_gravity = Quaternion(0, [0, 0, 9.8])  # gravity vector quaternion

    for j in range(12):
        q_Sigma_Y_i_j = Quaternion(Sigma_Y_i[0, j], Sigma_Y_i[1:4, j])
        q_mid = q_Sigma_Y_i_j.inv() * q_gravity * q_Sigma_Y_i_j
        Z[0:3, j] = q_mid.vec()
        Z[3:6, j] = Sigma_Y_i[4:7, j]

    return Z


def estimate_rot(data_num=1):
    # load data
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')

    accel = imu['vals'][0:3, :]
    gyro = imu['vals'][3:6, :]
    T = np.shape(imu['ts'])[1]

    ### Your code goes here

    ## IMU Calibration

    # # Load reference data
    # vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')  # not needed for autograder
    # T_vicon = np.shape(vicon['ts'])[1]
    #
    # # Accelerometer
    # # The magnitude of the acceleration should be close to 9.81 m/s^2
    # accel_mag = np.linalg.norm(accel, axis=0)
    # # Plot the magnitude of the acceleration
    # plt.figure()
    # plt.plot(accel_mag)
    # plt.title('IMU Acceleration Magnitude')
    #
    # # From the plot, we know that the data before 500 and after 4800 are stationary,
    # # so we can use them to calibrate the 9.81 m/s^2
    #
    # # Plot the IMU acceleration data for each axis
    # plt.figure()
    # plt.plot(accel[0, :])
    # plt.plot(accel[1, :])
    # plt.plot(accel[2, :])
    # plt.title('IMU Acceleration Data')
    # plt.legend(['x', 'y', 'z'])
    # plt.show()
    #
    # # Extract data 0-500 (AZ down)
    # accel_calib = accel[:, 0:500]
    # # Compute the mean of the acceleration for all axes
    # # The mean of AZ should be close to +9.81 m/s^2, while the mean of AX and AY should be close to 0
    # accel_mean = np.mean(accel_calib, axis=1)
    # print("AZ down: ")
    # print(accel_mean)
    #
    # # Extract data 1900-2000 (AX up)
    # accel_calib = accel[:, 1900:2000]
    # # Compute the mean of the acceleration for all axes
    # # The mean of AX should be close to -9.81 m/s^2, while the mean of AY and AZ should be close to 0
    # accel_mean = np.mean(accel_calib, axis=1)
    # print("AX up: ")
    # print(accel_mean)
    #
    # # Extract data 2550-2700 (AZ down)
    # accel_calib = accel[:, 2550:2700]
    # # Compute the mean of the acceleration for all axes
    # # The mean of AZ should be close to +9.81 m/s^2, while the mean of AX and AY should be close to 0
    # accel_mean = np.mean(accel_calib, axis=1)
    # print("AZ down: ")
    # print(accel_mean)
    #
    # # Extract data 3350-3420 (AY up)
    # accel_calib = accel[:, 3350:3420]
    # # Compute the mean of the acceleration for all axes
    # # The mean of AY should be close to -9.81 m/s^2, while the mean of AX and AZ should be close to 0
    # accel_mean = np.mean(accel_calib, axis=1)
    # print("AY up: ")
    # print(accel_mean)
    #
    # # Extract data 3870-3930 (AY down)
    # accel_calib = accel[:, 3870:3930]
    # # Compute the mean of the acceleration for all axes
    # # The mean of AY should be close to +9.81 m/s^2, while the mean of AX and AZ should be close to 0
    # accel_mean = np.mean(accel_calib, axis=1)
    # print("AY down: ")
    # print(accel_mean)
    #
    # # By manual calculation, the parameters are as below
    # alpha_ax = 34.9051
    # alpha_ay = (34.5250 + 33.1863) / 2  # 33.85565
    # alpha_az = 34.6828
    # beta_ax = 510.808
    # beta_ay = 500.994
    # beta_az = 499.69
    #
    # # Gyroscope
    # # The magnitude of the angular velocity should be close to 0 when stationary
    #
    # # The order of Gyro data is [Z, X, Y]
    # gyro_x = gyro[1, :]
    # gyro_y = gyro[2, :]
    # gyro_z = gyro[0, :]
    # # Plot the magnitude of the angular acceleration for each axis
    # plt.figure()
    # plt.plot(gyro_x)
    # plt.plot(gyro_y)
    # plt.plot(gyro_z)
    # plt.title('IMU Angular Acceleration Magnitude')
    # plt.legend(['x', 'y', 'z'])
    #
    # # Plot the vicon data
    # # vicon rots has shape (3, 3, 5561)
    # vicon_rots = vicon['rots']
    # vicon_roll = np.zeros(T_vicon)
    # vicon_pitch = np.zeros(T_vicon)
    # vicon_yaw = np.zeros(T_vicon)
    # for i in range(T_vicon):
    #     vicon_rot = vicon_rots[:, :, i]
    #     q = Quaternion()
    #     q.from_rotm(vicon_rot)
    #     euler = q.euler_angles()
    #     vicon_roll[i] = euler[0]
    #     vicon_pitch[i] = euler[1]
    #     vicon_yaw[i] = euler[2]
    # # Differentiate the vicon data using vicon timestamps to get the ground true angular velocity
    # vicon_roll_diff = np.zeros(T_vicon - 1)
    # vicon_pitch_diff = np.zeros(T_vicon - 1)
    # vicon_yaw_diff = np.zeros(T_vicon - 1)
    # for i in range(T_vicon - 1):
    #     vicon_roll_diff[i] = (vicon_roll[i + 1] - vicon_roll[i]) / (vicon['ts'][0, i + 1] - vicon['ts'][0, i])
    #     if vicon_roll_diff[i] > 10:  # Remove the jump between pi and -pi, as well as the noise
    #         vicon_roll_diff[i] = 0
    #     if vicon_roll_diff[i] < -10:
    #         vicon_roll_diff[i] = 0
    #     vicon_pitch_diff[i] = (vicon_pitch[i + 1] - vicon_pitch[i]) / (vicon['ts'][0, i + 1] - vicon['ts'][0, i])
    #     if vicon_pitch_diff[i] > 10:
    #         vicon_pitch_diff[i] = 0
    #     if vicon_pitch_diff[i] < -10:
    #         vicon_pitch_diff[i] = 0
    #     vicon_yaw_diff[i] = (vicon_yaw[i + 1] - vicon_yaw[i]) / (vicon['ts'][0, i + 1] - vicon['ts'][0, i])
    #     if vicon_yaw_diff[i] > 10:
    #         vicon_yaw_diff[i] = 0
    #     if vicon_yaw_diff[i] < -10:
    #         vicon_yaw_diff[i] = -0
    #
    # plt.figure()
    # plt.plot(vicon_roll_diff)
    # plt.plot(vicon_pitch_diff)
    # plt.plot(vicon_yaw_diff)
    # plt.title('Vicon Angular Acceleration')
    # plt.legend(['roll', 'pitch', 'yaw'])
    # plt.show()
    #
    # # The gyro bias can be estimated by averaging the gyro data when the IMU is stationary during time 0-500
    # beta_rx = np.mean(gyro_x[0:500])
    # beta_ry = np.mean(gyro_y[0:500])
    # beta_rz = np.mean(gyro_z[0:500])
    # print("Gyro Bias: ")
    # print([beta_rx, beta_ry, beta_rz])  # [373.568, 375.356, 369.68]
    #
    # beta_rx = 373.568
    # beta_ry = 375.356
    # beta_rz = 369.68
    #
    # # We can formulate a least squares problem to solve for the gyroscope sensitivity
    # # using the vicon data as ground truth
    # imu_rx_wo_bias = gyro_x - beta_rx
    # imu_ry_wo_bias = gyro_y - beta_ry
    # imu_rz_wo_bias = gyro_z - beta_rz
    #
    # S_x = solve_scale_factors(imu_rx_wo_bias, vicon_roll_diff)
    # S_y = solve_scale_factors(imu_ry_wo_bias, vicon_pitch_diff)
    # S_z = solve_scale_factors(imu_rz_wo_bias, vicon_yaw_diff)
    #
    # alpha_rx = 1 / (S_x * 1023 / 3300)
    # alpha_ry = 1 / (S_y * 1023 / 3300)
    # alpha_rz = 1 / (S_z * 1023 / 3300)
    # print("Gyro Sensitivity: ")
    # print([alpha_rx, alpha_ry, alpha_rz])  # [173.64053146945565, 204.52269156734403, 439.8366989965631]
    #
    # # The alpha_rz data may not be reliable, so we can use the following parameters
    # alpha_rx = 173.64053146945565
    # alpha_ry = 204.52269156734403
    # alpha_rz = 204.52269156734403  # Originally 439.8366989965631

    ## Unscented Kalman Filter

    # alpha_ax = 34.9051
    # alpha_ay = 33.85565
    # alpha_az = 34.6828
    # beta_ax = 510.808
    # beta_ay = 500.994
    # beta_az = 499.69
    alpha_ax = 33.85565
    alpha_ay = 33.85565
    alpha_az = 33.85565
    beta_ax = 510.808
    beta_ay = 500.994
    beta_az = 497.69

    beta_rx = 373.568
    beta_ry = 372.356
    beta_rz = 369.68
    alpha_rx = 207.64053
    alpha_ry = 205.52269
    alpha_rz = 212.52269

    # Ref: https://github.com/YugAjmera/Robot-Localization-and-Mapping
    # Initialize roll, pitch, yaw as numpy arrays of length T
    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)

    dt = imu['ts'][0, 1:] - imu['ts'][0, 0:-1]
    dt = np.hstack((0,dt))

    # Calibrate the IMU data
    ax = -compute_imu_data(accel[0, :], alpha_ax, beta_ax)
    ay = -compute_imu_data(accel[1, :], alpha_ay, beta_ay)
    az = compute_imu_data(accel[2, :], alpha_az, beta_az)

    rx = compute_imu_data(gyro[1, :], alpha_ry, beta_ry)
    ry = compute_imu_data(gyro[2, :], alpha_rx, beta_rx)
    rz = compute_imu_data(gyro[0, :], alpha_rz, beta_rz)

    # Plot the calibrated IMU data
    # plt.figure()
    # plt.plot(ax)
    # plt.plot(ay)
    # plt.plot(az)
    # plt.title('Calibrated IMU Acceleration Data')
    # plt.legend(['x', 'y', 'z'])
    # plt.figure()
    # plt.plot(rx)
    # plt.plot(ry)
    # plt.plot(rz)
    # plt.title('Calibrated IMU Angular Velocity Data')
    # plt.legend(['x', 'y', 'z'])
    # plt.show()

    # Initialize the state vector and covariance matrices
    x_km1 = np.zeros((7, 1))
    x_km1[0, 0] = 1
    P_km1 = np.eye(6)
    # Initialize the process noise and measurement noise covariance matrices
    Q = 0.1 * np.eye(6)  # process noise matrix (6,6)
    R = 0.1 * np.eye(6)  # measurement noise matrix (6,6)

    # For plotting
    q_mean = []
    q_cov = []
    euler_mean = []
    euler_cov = []
    gyro_rad_s = []
    vicon_rad = []


    for i in range(T):

        # Sigma points generation and transformation
        Sigma_X_i, Sigma_Y_i = sigma_points_gen_and_trans(x_km1, P_km1, Q, dt, i)

        # Compute the mean and covariance of the transformed sigma points
        Y_mean = compute_mean_Y(Sigma_Y_i, Sigma_X_i[0:4, 0])
        W_residual = compute_W_residual(Sigma_Y_i, Y_mean)
        # Y_cov = compute_Y_cov(W_residual)  ###
        Y_cov = (W_residual @ W_residual.T) / 12  # Eq (67)

        # Compute sigma points in the measurement space (Z)
        Z = compute_Z(Sigma_Y_i)
        Z_mean = np.mean(Z, axis=1).reshape(6, 1)
        # Use data from IMU as the observation
        Z_observe = np.zeros((6, 1))
        Z_observe[0:3, 0] = np.array([ax[i], ay[i], az[i]])
        Z_observe[3:6, 0] = np.array([rx[i], ry[i], rz[i]])

        # Compute the innovation and the Kalman gain
        P_zz = (Z - Z_mean) @ (Z - Z_mean).T / 12  # uncertainty of the predicted measurement, Eq (68)
        P_vv = P_zz + R  # covariance of the measurement, Eq (69)
        P_xz = W_residual @ (Z - Z_mean).T / 12  # cross correlation matrix, Eq (70)
        K_k = P_xz @ np.linalg.inv(P_vv)  # Kalman gain, Eq (72)
        P_k = Y_cov - K_k @ P_vv @ K_k.T  # covariance update, Eq (75)
        v_k = Z_observe - Z_mean  # innovation

        # Update the state and covariance
        # Note that quaternions can not be directly added
        # So first, convert the innovation with Kalman gain to a quaternion
        q_Kv = Quaternion()
        v_k = K_k @ v_k
        q_Kv.from_axis_angle(v_k[0:3, 0])
        q_xk = Quaternion(Y_mean[0, 0], Y_mean[1:4, 0])
        # Then, multiply the quaternion with the state quaternion
        q_xk = q_Kv * q_xk
        q_xk.normalize()
        # Update the state vector
        x_km1[0:4, 0] = q_xk.q
        # Update the angular velocity, can be directly added
        x_km1[4:7, 0] += v_k[3:6, 0]

        # Update the covariance matrix
        P_km1 = P_k

        # Compute the roll, pitch, and yaw
        q_xk = Quaternion(x_km1[0, 0], x_km1[1:4, 0])
        euler = q_xk.euler_angles()
        roll[i] = euler[0]
        pitch[i] = euler[1]
        yaw[i] = euler[2]

        # Write into plotting variables
        q_mean.append(x_km1[0:4, 0].copy())
        q_cov.append(np.linalg.det(P_km1))
        euler_mean.append(euler.copy())
        # omg_cov.append()
        gyro_rad_s.append(Z_observe[3:6, 0].copy())
        # vicon_rad.append()

    return roll, pitch, yaw, q_mean, q_cov, euler_mean, gyro_rad_s


# Test
if __name__ == "__main__":
    for i in range(1, 2):
        roll, pitch, yaw, q_mean, q_cov, euler_mean, gyro_rad_s = estimate_rot(i)
        print("Roll: ", roll)
        print("Pitch: ", pitch)
        print("Yaw: ", yaw)

        plt.figure()
        plt.plot(q_mean)  # 4D
        plt.legend(["q0", "q1", "q2", "q3"])
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("$q_{mean}$")

        plt.figure()
        plt.xscale("log")
        # As we use log axis, we need to offset the indices by 1 so that the first valid data has index 1
        q_cov = np.insert(q_cov, 0, 0)
        plt.plot(q_cov)  # 6D
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("$q_{cov}$")

        plt.figure()
        plt.plot(euler_mean)
        plt.legend(["row", "pitch", "yaw"])
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("Mean of Euler Angles")


        plt.figure()
        plt.plot(gyro_rad_s)
        # plt.legend(["q0", "q1", "q2", "q3"])
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("Gyroscope rotation speed")

        plt.show()
