"""
Extended Kalman Filter (EKF) Localization using IMU + Visual Odometry (VO)
- Blue line: Ground truth
- Black line: Dead reckoning
- Green points: VO measurements
- Red line: EKF estimate
- Red ellipse: Covariance
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
DT = 0.01         # [s] time step
SIM_TIME = 40.0   # [s] simulation time
show_animation = True

# ---------- Noise Covariances (stds -> squared to variances) ----------
Q = np.diag([ #7x7
    1e-3, 1e-3, np.deg2rad(0.05),   # pos x, pos y, yaw (std)
    5e-3, 5e-3,                     # vx, vy (std)
    np.deg2rad(0.01),               # gyro bias (std)
    1e-2, 1e-2                      # accel bias x,y (std)
]) ** 2

R = np.diag([ #3x3
    0.05, 0.05, np.deg2rad(1.0)     # VO measurement noise (std)
]) ** 2

GYRO_NOISE   = np.deg2rad(0.3)
ACC_NOISE_XY = 0.05
BIAS_RW_G    = np.deg2rad(0.005)
BIAS_RW_A    = 0.002

# ---------------- Helpers ----------------
def wrap_angle(a): # Wrap from -180 to 180
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2d(theta): # Finds the rotation 
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# ---------------- Motion Model ----------------
def motion_model(x, u):
    """
    x: [x, y, yaw, vx, vy, bg, bax, bay] (8x1)    For IMU
    u: [omega_meas, ax_body_meas, ay_body_meas] (3x1)     For visual
    """
    px, py, yaw, vx, vy, bg, bax, bay = x.flatten()
    omega_m, ax_b_m, ay_b_m = u.flatten()

    # Debias IMU, find true values by subrating out the bias
    omega = omega_m - bg
    a_b = np.array([ax_b_m - bax, ay_b_m - bay]).reshape(2,1)

    # Rotate accel to world frame, just shift the acceleration vector of the robot to the world frame
    Rbw = rot2d(yaw)
    a_w = (Rbw @ a_b).flatten()

    # Integrate (simple constant-accel)
    vx_new = vx + a_w[0]*DT
    vy_new = vy + a_w[1]*DT
    px_new = px + vx*DT + 0.5*a_w[0]*DT*DT
    py_new = py + vy*DT + 0.5*a_w[1]*DT*DT
    yaw_new = wrap_angle(yaw + omega*DT)

    return np.array([[px_new, py_new, yaw_new, vx_new, vy_new, bg, bax, bay]]).T

def observation_model(x): # Reshapes the inputs for proper matrix math
    # VO measures pose: [x, y, yaw]
    H = np.zeros((3, 8))
    H[0,0] = 1.0
    H[1,1] = 1.0
    H[2,2] = 1.0
    return H @ x

# ---------------- Jacobians (numerical for f only) ----------------
def numerical_jacobian_f(f, x, u, eps=1e-6):
    # u is the control matrix, so the model can adjust to the control system too
    # x is the state matrix
    # eps is just the step size
    n = x.shape[0] # number of states 8
    Fx = np.zeros((n, n)) # 8x8 zero matrix
    fx = f(x, u) # Calls the motion model with x and u, python takes the function as a parameter
    for i in range(n): # loop through all the i 
        dx = np.zeros((n,1)); dx[i,0] = eps # Format the step size matrix
        Fx[:, [i]] = (f(x+dx, u) - fx)/eps # solves for whole column
    return Fx # Return the jacobian

def H_analytical(): # simply maps the measurement matrix H to do math later
    H = np.zeros((3, 8))
    H[0,0] = H[1,1] = H[2,2] = 1.0
    return H

# ---------------- EKF Core ----------------
def ekf_predict(xEst, PEst, u): # Predict function
    xPred = motion_model(xEst, u) # predict using motion model
    F = numerical_jacobian_f(motion_model, xEst, u) #Solve for jacobian
    PPred = F @ PEst @ F.T + Q # Predicted the next covariance
    # enforce symmetry numerically
    PPred = 0.5*(PPred + PPred.T) # fix rounding errors to make sure it is symmetrical
    return xPred, PPred # spit back the predictions

def ekf_update(xPred, PPred, z): # Update the values
    H = H_analytical() # get the measurement matrix
    zPred = observation_model(xPred) # Take in the measurements
    y = z - zPred # Find the residual
    y[2,0] = wrap_angle(y[2,0]) # wrap the angle if its outside of range

    S = H @ PPred @ H.T + R # Find the total covariance
    K = PPred @ H.T @ np.linalg.inv(S) # Calculate the Kalman Gain

    xUpd = xPred + K @ y # Updated measurement
    I = np.eye(PPred.shape[0]) # Identity matrix
    # Joseph form for numerical stability + PSD
    PUpd = (I - K @ H) @ PPred @ (I - K @ H).T + K @ R @ K.T # Update the covariance
    # enforce symmetry
    PUpd = 0.5*(PUpd + PUpd.T)

    xUpd[2,0] = wrap_angle(xUpd[2,0]) # wrap the angle if needed
    return xUpd, PUpd # return the updated

# ---------------- Simulation Models ----------------
def calc_truth_input(t): # Initalize the starting conditions
    # commanded forward speed and yawrate (simple demo)
    v = 1.2
    yawrate = 0.15
    ax_body_true = 0.0
    ay_body_true = v * yawrate   # centripetal (body y)
    return v, yawrate, ax_body_true, ay_body_true

def propagate_truth(xTrue, v, yawrate): #Simulate the actual movement
    px, py, yaw, vx, vy = xTrue.flatten()
    yaw_new = wrap_angle(yaw + yawrate*DT) # wrap angle
    vx_new = v*math.cos(yaw_new) # calculate the velocities
    vy_new = v*math.sin(yaw_new)
    # trapezoidal integration for position (less phasing)
    px_new = px + 0.5*(vx + vx_new)*DT
    py_new = py + 0.5*(vy + vy_new)*DT
    return np.array([[px_new, py_new, yaw_new, vx_new, vy_new]]).T # return the new values

def imu_measure(true_biases, yaw, ax_b_true, ay_b_true, yawrate_true):
    bg, bax, bay = true_biases # get bias
    omega_m = yawrate_true + bg + np.random.randn()*GYRO_NOISE # simulate the imu data
    ax_m = ax_b_true + bax + np.random.randn()*ACC_NOISE_XY
    ay_m = ay_b_true + bay + np.random.randn()*ACC_NOISE_XY
    return np.array([[omega_m, ax_m, ay_m]]).T

def drift_biases(true_biases): #Simulates the drift of biases over time
    bg, bax, bay = true_biases
    bg  += np.random.randn()*BIAS_RW_G
    bax += np.random.randn()*BIAS_RW_A
    bay += np.random.randn()*BIAS_RW_A
    return (bg, bax, bay)

def vo_measure(xTrue): # simulates the visual odometry sensor measures 
    z = xTrue[[0,1,2], :] + np.array([[np.random.randn()*np.sqrt(R[0,0])],
                                      [np.random.randn()*np.sqrt(R[1,1])],
                                      [np.random.randn()*np.sqrt(R[2,2])]])
    z[2,0] = wrap_angle(z[2,0])
    return z

# ---------------- Covariance Ellipse ----------------
def plot_covariance_ellipse(x, y, P, nsig=2.0):
    # P should be 2x2
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.clip(eigvals, 0, None)
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(t), np.sin(t)))  # 2xN
    ellipse = eigvecs @ (np.sqrt(eigvals)[:, None] * circle) * nsig
    ex = x + ellipse[0, :]
    ey = y + ellipse[1, :]
    plt.plot(ex, ey, "--r", alpha=0.6)

# ---------------- Main ----------------
def main():
    # States
    xEst = np.zeros((8,1))
    PEst = np.eye(8)*1e-2
    xTrue_posevel = np.zeros((5,1))  # [x, y, yaw, vx, vy]
    xDR = np.zeros((8,1))            # Dead reckoning state (IMU-only)
    true_biases = (0.0, 0.0, 0.0)

    # --- Important: give truth & dead-reckoning a sensible initial velocity ---
    v0, yawrate0, ax0, ay0 = calc_truth_input(0.0)
    # assume initial yaw = 0 in both truth and DR; set initial vx,vy to commanded v
    xTrue_posevel[3,0] = v0 * math.cos(xTrue_posevel[2,0])
    xTrue_posevel[4,0] = v0 * math.sin(xTrue_posevel[2,0])
    xDR[3,0] = xTrue_posevel[3,0]
    xDR[4,0] = xTrue_posevel[4,0]

    # Logs
    hxEst = xEst
    # append biases (zeros) to the initial truth log to match dimensions
    hxTrue = np.vstack((xTrue_posevel, np.zeros((3,1))))
    hxDR = xDR
    hz = observation_model(xEst)

    t = 0.0
    while t <= SIM_TIME:
        t += DT

        # Ground truth
        v, yawrate, ax_b_true, ay_b_true = calc_truth_input(t)
        xTrue_posevel = propagate_truth(xTrue_posevel, v, yawrate)
        true_biases = drift_biases(true_biases)

        # IMU measurement (from true motion + true biases)
        u_m = imu_measure(true_biases, xTrue_posevel[2,0], ax_b_true, ay_b_true, yawrate)

        # Dead reckoning (IMU only, no VO correction)
        xDR = motion_model(xDR, u_m)

        # EKF predict & update
        xPred, PPred = ekf_predict(xEst, PEst, u_m)
        z = vo_measure(xTrue_posevel)
        xEst, PEst = ekf_update(xPred, PPred, z)

        # Log
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, np.vstack((xTrue_posevel,
                                               np.array([[true_biases[0], true_biases[1], true_biases[2]]]).T))))
        hz = np.hstack((hz, z))

        # Visualization
        if show_animation and int(t/DT) % 5 == 0:
            plt.cla()
            plt.plot(hz[0, :], hz[1, :], ".g", label="VO meas")
            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b", label="Ground truth")
            plt.plot(hxDR[0, :], hxDR[1, :], "-k", label="Dead reckoning")
            plt.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF estimate")
            plot_covariance_ellipse(xEst[0,0], xEst[1,0], PEst[0:2,0:2], nsig=2)
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

if __name__ == "__main__":
    main()