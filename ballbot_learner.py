import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
import time
import pickle
from replay_memory import ReplayMemory
import os

# ugly workaround until shared library can be discovered properly with python3
import sys
sys.path.append(os.environ["HOME"]+"/catkin_ws/devel/lib/python3.6/dist-packages/ocs2_ballbot_example")

from BallbotPyBindings import mpc_interface, scalar_array, state_vector_array, input_vector_array, dynamic_vector_array, cost_desired_trajectories

from PolicyNet import ExpertMixturePolicy as PolicyNet

mpc = mpc_interface("mpc", False)
systemHasConstraints = False

def getTargetTrajectories():
    desiredTimeTraj = scalar_array()
    desiredTimeTraj.resize(1)
    desiredTimeTraj[0] = 2.0

    desiredInputTraj = dynamic_vector_array()
    desiredInputTraj.resize(1)
    desiredInputTraj[0] = np.zeros((mpc.INPUT_DIM, 1))

    desiredStateTraj = dynamic_vector_array()
    desiredStateTraj.resize(1)
    desiredStateTraj[0] = np.zeros((mpc.STATE_DIM, 1))

    return cost_desired_trajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)


targetTrajectories = getTargetTrajectories()

mpc.reset(targetTrajectories)



dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU



class FlowMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        xDot = torch.tensor(mpc.computeFlowMap(t, x_np, u_np), device=device, dtype=dtype)
        return xDot

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of dynamics w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            mpc.setFlowMapDerivativeStateAndControl(t, x_np, u_np)
            dfdx = torch.tensor(mpc.computeFlowMapDerivativeState(), device=device, dtype=dtype)
            grad_x = torch.matmul(grad_output, dfdx).reshape((-1, x_np.size))
        if ctx.needs_input_grad[2]:
            mpc.setFlowMapDerivativeStateAndControl(t, x_np, u_np)
            dfdu = torch.tensor(mpc.computeFlowMapDerivativeInput(), device=device, dtype=dtype)
            grad_u = torch.matmul(grad_output, dfdu).reshape((-1, u_np.size))
        return grad_t, grad_x, grad_u


class IntermediateCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        L = torch.tensor(mpc.getIntermediateCost(t, x_np, u_np), device=device, dtype=dtype)
        return L

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of RunningCost w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            dLdx = torch.tensor([[mpc.getIntermediateCostDerivativeState(t, x_np, u_np)]], device=device, dtype=dtype)
            grad_x = grad_output * dLdx
        if ctx.needs_input_grad[2]:
            dLdu = torch.tensor([[mpc.getIntermediateCostDerivativeInput(t, x_np, u_np)]], device=device, dtype=dtype)
            grad_u = grad_output * dLdu
        return grad_t, grad_x, grad_u


class StateInputConstraint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        g1 = torch.tensor(mpc.getStateInputConstraint(t, x_np, u_np), device=device, dtype=dtype)
        return g1

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of StateInputConstraint w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("Derivative of StateInputConstraint w.r.t. state not available")
        if ctx.needs_input_grad[2]:
            dg1du = torch.tensor(mpc.getStateInputConstraintDerivativeControl(t, x_np, u_np), device=device, dtype=dtype)
            grad_u = torch.matmul(grad_output, dg1du).reshape((-1, u_np.size))
        return grad_t, grad_x, grad_u


def loss_function(tx, u_pred, dVdx, nu):
    f = FlowMap.apply(tx[0], tx[1:], u_pred)
    L = IntermediateCost.apply(tx[0], tx[1:], u_pred)
    loss = L + dVdx.dot(f)
    if systemHasConstraints:
        g1 = StateInputConstraint.apply(tx[0], tx[1:], u_pred)
        loss += g1.dot(nu)
    return loss


def num_samples_per_trajectory_point(t, max_num_points, half_value_decay_t):
    """
    Calculates number of samples drawn for each nominal state point in trajectory
    :param t: Query time along trajectory
    :param max_num_points:
    :param half_value_decay_t: time into trajectory after which number of sampled point is halfed
    :return: Number of samples to be drawn
    """
    return max_num_points * np.exp(-np.log(2) * t / half_value_decay_t)


def trajectoryCost(policy, duration, dt_control):
    cost = 0.0  # running sum
    numStartingPoints = 1
    for _ in range(numStartingPoints):
        startPos = np.zeros([mpc.STATE_DIM, 1])
        tx = np.concatenate(([[0.0]], startPos))
        for it in range(int(duration / dt_control)):
            ttx_torch = torch.tensor(np.concatenate((tx[0, 0], tx[1:]), axis=None), dtype=dtype,
                                   device=device, requires_grad=False)
            p, u_pred = policy(ttx_torch)
            if len(p) > 1:
                u = torch.matmul(p, u_pred)
            else:
                u = u_pred[0]

            u_np = u.t().detach().numpy().astype('float64')
            cost += torch.tensor(mpc.getIntermediateCost(tx[0], tx[1:], u_np), device=device, dtype=dtype)
            if np.isnan(cost):
                return np.nan, tx[0]
            dx = mpc.computeFlowMap(tx[0], tx[1:], u_np)

            tx[1:] += dx.reshape(mpc.STATE_DIM, 1) * dt_control
            tx[0, 0] += dt_control
    return cost, duration


writer = SummaryWriter()


load_policy = False
if load_policy:
    save_path = "/path/to/saved/policy.pt"
    policy = torch.load(save_path)
    policy.eval()
else:
    policy = PolicyNet(mpc.STATE_DIM+1, mpc.INPUT_DIM)

policy.to(device)

print("Initial policy parameters:")
print(list(policy.named_parameters()))

learning_rate = 1e-2
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


load_memory = False
if load_memory:
    with open("/path/to/memory.pkl", 'rb') as memFile:
        mem = pickle.load(memFile)
else:
    mem_capacity = 1000000
    mem = ReplayMemory(mem_capacity)

# prepare saving of MPC solution trajectory (always add first point of a slq run)
mpc_traj_len_sec = 3.0 # length of trajectories to generate with MPC
dt_control = 1.0/400. # 400 Hz control frequency
mpc_traj_t = np.linspace(0.0, mpc_traj_len_sec, int(mpc_traj_len_sec/dt_control))
last_policy_save_time = time.time()

learning_iterations = 100000

print("==============\nStarting training\n==============")
try:
    for it in range(learning_iterations):
        alpha_mix = np.clip(1.0 - 1.0 * it / learning_iterations, 0.2, 1.0)

        # run data collection (=MPC) less frequently than the policy updates
        mpc_decimation = 1 if len(mem) < 15000 else 500
        if it % mpc_decimation == 0:

            mpc.reset(targetTrajectories)
            x0 = np.zeros((mpc.STATE_DIM, 1))
            x0[0] = np.random.uniform(-0.5, 0.5) # base x
            x0[1] = np.random.uniform(-0.5, 0.5) # base y


            print("resetting MPC")
            print("proportion of MPC policy is", alpha_mix)
            print("starting from", x0.transpose())

            for mpc_time in mpc_traj_t: # mpc dummy loop
                mpc.setObservation(mpc_time, x0)
                try:
                    mpc.advanceMpc()
                except RuntimeError:
                    print("Caught error in MPC advance!!")
                    break
                t_result = scalar_array()
                x_result = state_vector_array()
                u_result = input_vector_array()
                mpc.getMpcSolution(t_result, x_result, u_result)
                K = mpc.getLinearFeedbackGain(t_result[0])
                # sample around the initial point and push it to the replay buffer
                for i in range(int(round(num_samples_per_trajectory_point(t_result[0], max_num_points=4, half_value_decay_t=1e10)))):
                    if i == 0:
                        x = x_result[0] # definitely push back the nominal point
                    else:
                        x = np.random.multivariate_normal(x_result[0], cov=np.diag(0.1 * np.ones(mpc.STATE_DIM)))

                    dVdx = mpc.getValueFunctionStateDerivative(t_result[0], x)
                    if systemHasConstraints:
                        nu = mpc.getStateInputConstraintLagrangian(t_result[0], x)
                    else:
                        nu = None
                    mem.push(mpc_time, x, dVdx, None, nu, None, u_result[0] + K.dot(x - x_result[0]))

                # increment state for next time step
                ttx_torch = torch.tensor(np.concatenate((t_result[0], x_result[0]), axis=None),
                                         dtype=torch.float, requires_grad=False)
                p, u_net = policy(ttx_torch)
                if len(p) > 1:
                    u_net = torch.matmul(p, u_net)
                else:
                    u_net = u_net[0]

                u_mixed = alpha_mix * u_result[0] + (1.0 - alpha_mix) * u_net.detach().numpy().astype('float64')
                dx = mpc.computeFlowMap(t_result[0], x_result[0], u_mixed)
                x0 += dt_control * dx.reshape(mpc.STATE_DIM,1)

            print("mpc ended up at", x_result[0])

        # extract batch of samples from replay memory
        batch_size = 2**5
        samples = mem.sample(batch_size)

        writeLogThisIteration = True

        def solver_step_closure():
            loss = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            mpc_H = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            g1_norm = 0.0  # running sum over samples
            for sample in samples:
                tx = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                ttx_net = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                p, u_pred = policy(ttx_net)
                dVdx = torch.tensor(sample.dVdx, dtype=dtype, device=device, requires_grad=False)
                if systemHasConstraints:
                    nu = torch.tensor(sample.nu, dtype=dtype, device=device, requires_grad=False)
                else:
                    nu = None
                for pi, u_pred_i in zip(p, u_pred): # loop through experts
                    loss += pi * loss_function(tx, u_pred_i, dVdx, nu)
                mpc_H += loss_function(tx, torch.tensor(sample.u0), dVdx, nu)

                if len(p) > 1:
                    u_net = torch.matmul(p, u_pred)
                else:
                    u_net = u_pred[0]

                if systemHasConstraints:
                    g1_norm += np.linalg.norm(mpc.getStateInputConstraint(sample.t, sample.x, u_net.detach().numpy().astype('float64')))

            optimizer.zero_grad()
            loss.backward()

            global writeLogThisIteration
            if writeLogThisIteration:
                writer.add_scalar('loss/perSample', loss.item() / batch_size, it)
                writer.add_scalar('loss/constraintViolation', g1_norm / batch_size, it)
                writeLogThisIteration = False

            return loss


        if it % 200 == 0:
            oc_cost, survival_time = trajectoryCost(policy=policy, duration=mpc_traj_len_sec, dt_control=dt_control)
            writer.add_scalar('metric/oc_cost', oc_cost, it)
            writer.add_scalar('metric/survival_time', survival_time, it)
            print("iteration", it, "oc_cost", oc_cost)

        if time.time() - last_policy_save_time > 5.0 * 60.0:
            last_policy_save_time = time.time()
            now = datetime.datetime.now()
            save_path = "/tmp/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
            print("Iteration", it, "saving policy to", save_path + ".pt")
            torch.save(policy, save_path + ".pt")


        optimizer.step(solver_step_closure)
        for param in policy.parameters():
            if(torch.isnan(param).any()):
                print("nan in policy!")


    print("==============\nTraining completed.\n==============")
except KeyboardInterrupt:
    print("==============\nTraining interrupted after iteration", it, ".\n==============")
    pass



print("optimized policy parameters:")
print(list(policy.named_parameters()))

now = datetime.datetime.now()
save_path = "/tmp/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
print("saving policy to", save_path + ".pt")
torch.save(policy, save_path + ".pt")

# print("Saving data to", save_path+"_memory.pkl")
# with open(save_path+"_memory.pkl", 'wb') as outputFile:
#     pickle.dump(mem, outputFile)


writer.close()

print("Done. Exiting now.")
