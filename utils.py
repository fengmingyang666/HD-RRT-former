import numpy as np


def compute_jerk_value(q):
    """
    q: np.array of shape (T, D)
    return: scalar jerk measure
    """
    jerk = q[3:] - 3*q[2:-1] + 3*q[1:-2] - q[:-3]   # (T-3, D)
    return float(np.sum(np.linalg.norm(jerk, axis=1)**2))


def compute_curvature_value(q):
    """
    q: np.array of shape (T, D)
    return: scalar curvature measure
    """
    curv = q[2:] - 2*q[1:-1] + q[:-2]   # (T-2, D)
    return float(np.sum(np.linalg.norm(curv, axis=1)**2))


def compute_energy_value(q, alpha=0.1):
    """
    q: np.array of shape (T, D)
    return: scalar energy: sum(vel^2 + alpha * acc^2)
    """
    vel = q[1:] - q[:-1]                    # (T-1, D)
    acc = q[2:] - 2*q[1:-1] + q[:-2]        # (T-2, D)

    vel_energy = np.sum(np.linalg.norm(vel, axis=1)**2)
    acc_energy = np.sum(np.linalg.norm(acc, axis=1)**2)

    return float(vel_energy + alpha * acc_energy)

def compute_acceleration_value(q):
    """
    q: np.array of shape (T, D)
    return: scalar acceleration measure
    """
    # second derivative (acceleration)
    acc = q[2:] - 2*q[1:-1] + q[:-2]    # (T-2, D)

    # sum of squared L2 norms across time
    return float(np.sum(np.linalg.norm(acc, axis=1)**2))
