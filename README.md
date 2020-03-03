# MPC-Net

This package contains supplementary code and implementation details for the [publication](https://doi.org/10.1109/LRA.2020.2974653)
> J. Carius, F. Farshidian and M. Hutter, "MPC-Net: A First Principles Guided Policy Search," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2897-2904, April 2020.

A preprint is available on [arxiv](https://arxiv.org/pdf/1909.05197.pdf).

While licensing restrictions do not allow us to release the ANYmal model,
we are providing our training script with an alternative ball-balancing robot.

## Dependencies
 * [OCS2 Toolbox](https://bitbucket.org/leggedrobotics/ocs2/)
 * [Pybind11](https://github.com/pybind/pybind11)
 * [Pytorch](https://pytorch.org/)
 * [TensorboardX](https://pypi.org/project/tensorboardX/)
 * [Matplotlib](https://matplotlib.org/)

## Setup Instructions
* Build and install Pybind11 according to the instructions in their documentation.
Make sure CMake can locate the Pybind11 installation, for example by adding the install path to your `CMAKE_PREFIX_PATH`.

* Clone [OCS2](https://bitbucket.org/leggedrobotics/ocs2/) into the source folder of a catkin workspace.
Then build the python bindings for the optimal control solver with<br>
`catkin build ocs2_ballbot_example --cmake-args -DUSE_PYBIND_PYTHON_3=ON`
* Install required python packages<br>
`pip3 install torch tensorboardX matplotlib`<br>
Note that we use python3 as it is required for pytorch.

## Running the Policy Training
Make sure your catkin workspace is sourced in the current terminal.
The policy training can then be started with the command<br>
`python3 ballbot_learner.py`

To monitor progress, execute tensorboard<br>
`tensorboard --logdir runs`

During training, the policy will be saved to disk in regular intervals.
The performance of the policy on the internal model can be visualized by running the script<br>
`python3 ballbot_evaluation.py`
