import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.environ["HOME"]+"/catkin_ws/devel/lib/python3.6/dist-packages/ocs2_ballbot_example")

from BallbotPyBindings import mpc_interface, scalar_array, state_vector_array, input_vector_array, dynamic_vector_array, cost_desired_trajectories

mpc = mpc_interface("mpc", False)


def plot(save_path, t_end=10.0):
    policy = torch.load(save_path)

    dt = 1./400.
    tx0 = np.zeros((mpc.STATE_DIM + 1, 1))
    tx0[1, 0] = -0.5
    tx0[2, 0] = 0.5

    tx_history = np.zeros((int(t_end/dt)+1, mpc.STATE_DIM + 1))

    tx = tx0
    average_constraint_violation = 0
    steps = int(t_end/dt)

    for it in range(steps):
        tx_history[it, :] = np.transpose(tx)
        tx_torch = torch.tensor(np.transpose(tx), dtype=torch.float, requires_grad=False)
        tx_torch[0][0] = 0.0 #optionally run it in MPC style

        p, u_pred = policy(tx_torch)
        if len(p) > 1:
            u = torch.mm(p.t(), u_pred)
        else:
            u = u_pred[0]

        u_np = u.t().detach().numpy().astype('float64')

        dx = mpc.computeFlowMap(tx[0], tx[1:], u_np)

        np.transpose(tx[1:])[0] += dx*dt
        tx[0] += dt

    tx_history[it+1,:] = np.transpose(tx)

    plt.figure(figsize=(8, 8))
    lineObjects = plt.plot(tx_history[:, 0], tx_history[:, 1:6])
    plt.legend(iter(lineObjects), ('px', 'py','thetaz','thetay','thetax'))




plot(save_path="/path/to/saved/policy.pt", t_end=10.0)
plt.show()
