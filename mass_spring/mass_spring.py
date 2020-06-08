import random
import sys
import taichi as tc
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
from mass_spring_robot_config import robots
from scipy.ndimage.filters import gaussian_filter


### INITIALIZATION ###
np.random.seed(0)
random.seed(0)
ti.init(arch=ti.gpu, default_fp=ti.f32)


### DATA TYPE SHORTCUTS ###
real   = ti.f32
scalar = lambda: ti.var(dt=real)
vec    = lambda: ti.Vector(2, dt=real)


### ROBOT CONFIGURATION (will be sinked down below) ###
n_objects = 0
n_springs = 0
def setup_robot(objects, springs):
  global n_objects, n_springs
  n_objects = len(objects)
  n_springs = len(springs)

  print(n_objects)

  print('n_objects=', n_objects, '   n_springs=', n_springs)

  for i in range(n_objects):
    x[0, i] = objects[i]

  for i in range(n_springs):
    s = springs[i]
    spring_anchor_a[i] = s[0]
    spring_anchor_b[i] = s[1]
    spring_length[i] = s[2]
    spring_stiffness[i] = s[3]
    spring_actuation[i] = s[4]


### SIMULATION PARAMETERS ###
GROUND_HEIGHT = 0.1
GRAVITY = -4.8
MAX_STEPS = 4096
dt = 0.004


### POLICY HYPERPARAMETERS ###
N_SIN_WAVES = 10
N_HIDDEN = 32


### TRAINING HYPERPARAMETERS ###
SIMULATION_TRAINING_STEPS = 2048 // 3
TRAINING_EPOCHS = 100
LEARNING_RATE   = 0.1
USE_TIME_OF_IMPACT = True
SPRING_OMEGA = 10
DAMPING      = 15


### DATA ###
loss  = scalar()
x     = vec()
v     = vec()
v_inc = vec()
spring_anchor_a = ti.var(ti.i32)
spring_anchor_b = ti.var(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()
weights1 = scalar()
bias1 = scalar()
weights2 = scalar()
bias2 = scalar()
hidden = scalar()
center = vec()
act = scalar()

HEAD_ID = 0


def n_input_states():
  return N_SIN_WAVES + 4 * n_objects + 2


### DATA LAYOUT CONFIGURATION (for faster computation) ###
@ti.layout
def place():
  ti.root.dense(ti.l, MAX_STEPS).dense(ti.i, n_objects).place(x, v, v_inc)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length, spring_stiffness,
                                       spring_actuation)
  ti.root.dense(ti.ij, (N_HIDDEN, n_input_states())).place(weights1)
  ti.root.dense(ti.i, N_HIDDEN).place(bias1)
  ti.root.dense(ti.ij, (n_springs, N_HIDDEN)).place(weights2)
  ti.root.dense(ti.i, n_springs).place(bias2)
  ti.root.dense(ti.ij, (MAX_STEPS, N_HIDDEN)).place(hidden)
  ti.root.dense(ti.ij, (MAX_STEPS, n_springs)).place(act)
  ti.root.dense(ti.i, MAX_STEPS).place(center)
  ti.root.place(loss)

  # Add gradient variables
  ti.root.lazy_grad()


### ENVIRONMENT SYSTEMS ###
@ti.kernel
def compute_center_of_mass(t: ti.i32):
  for _ in range(1):
    c = ti.Vector([0.0, 0.0])
    for i in ti.static(range(n_objects)):
      c += x[t, i]
    center[t] = (1.0 / n_objects) * c

@ti.kernel
def apply_spring_force(t: ti.i32):
  for i in range(n_springs):
    a = spring_anchor_a[i]
    b = spring_anchor_b[i]
    pos_a = x[t, a]
    pos_b = x[t, b]
    dist = pos_a - pos_b
    length = dist.norm() + 1e-4

    target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[t, i])
    # aka force over time
    impulse = dt * (length - target_length) * spring_stiffness[i] / length * dist

    ti.atomic_add(v_inc[t + 1, a], -impulse)
    ti.atomic_add(v_inc[t + 1, b], impulse)

@ti.kernel
def advance_toi(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * DAMPING)
    old_v = s * v[t - 1, i] + dt * GRAVITY * ti.Vector([0.0, 1.0]) + v_inc[t, i]
    old_x = x[t - 1, i]
    new_x = old_x + dt * old_v
    toi = 0.0
    new_v = old_v
    if new_x[1] < GROUND_HEIGHT and old_v[1] < -1e-4:
      toi = -(old_x[1] - GROUND_HEIGHT) / old_v[1]
      new_v = ti.Vector([0.0, 0.0])
    new_x = old_x + toi * old_v + (dt - toi) * new_v

    v[t, i] = new_v
    x[t, i] = new_x

@ti.kernel
def advance_no_toi(t: ti.i32):
  for i in range(n_objects):
    # Semi-Implicit Euler
    s = math.exp(-dt * DAMPING)
    old_v = s * v[t - 1, i] + dt * GRAVITY * ti.Vector([0.0, 1.0]) + v_inc[t, i]
    old_x = x[t - 1, i]
    new_v = old_v

    # Ground collision detection
    depth = old_x[1] - GROUND_HEIGHT
    if depth < 0 and new_v[1] < 0:
      # friction projection
      new_v[0] = 0
      new_v[1] = 0

    # Semi-Implicit Euler Cont.
    new_x = old_x + dt * new_v
    v[t, i] = new_v
    x[t, i] = new_x


### POLICY SYSTEMS ###
@ti.kernel
def policy_compute_layer1(t: ti.i32):
  """
  Input:
    - Current dt in terms of 10 sinusoids
    - Offset of every object with respect to the center
    - Velocity of every object
  Notes
    - Without current deltatime in terms of sinusoids - gradients explode
    - Without multiplication to 0.05 - gradients explode again
  """
  for i in range(N_HIDDEN):
    actuation = 0.0

    # Sinusoids
    for j in ti.static(range(N_SIN_WAVES)):
     actuation += weights1[i, j] * ti.sin(SPRING_OMEGA * t * dt +
                                         2 * math.pi / N_SIN_WAVES * j)

    # Objects
    for j in ti.static(range(n_objects)):
      offset = x[t, j] - center[t]

      # use a smaller weight since there are too many of them
      actuation += weights1[i, j * 4 + N_SIN_WAVES] * offset[0] * 0.05
      actuation += weights1[i, j * 4 + N_SIN_WAVES + 1] * offset[1] * 0.05
      actuation += weights1[i, j * 4 + N_SIN_WAVES + 2] * v[t, j][0] * 0.05
      actuation += weights1[i, j * 4 + N_SIN_WAVES + 3] * v[t, j][1] * 0.05
    
    actuation += bias1[i]
    actuation = ti.tanh(actuation)
    hidden[t, i] = actuation

@ti.kernel
def policy_compute_layer2(t: ti.i32):
  for i in range(n_springs):
    actuation = 0.0
    for j in ti.static(range(N_HIDDEN)):
      actuation += weights2[i, j] * hidden[t, j]
    actuation += bias2[i]
    actuation = ti.tanh(actuation)
    act[t, i] = actuation

def policy_init_network():
  """
    Xavier initialization of the policy network
  """
  for i in range(N_HIDDEN):
    for j in range(n_input_states()):
      # Xavier Normal Init
      gain = 1.0
      std  = math.sqrt(2 / float(N_HIDDEN + n_input_states())) * gain
      weights1[i, j] = np.random.randn() * std

  for i in range(n_springs):
    for j in range(N_HIDDEN):
      # Xavier Normal Init
      gain = 1.0
      std  = math.sqrt(2 / float(N_HIDDEN + n_springs)) * gain
      weights2[i, j] = np.random.randn() * std


### VISUALISATION SYSTEMS ###
VIS_INTERVAL = 128
gui          = tc.core.GUI("Mass Spring Robot", tc.veci(1024, 1024))
canvas       = gui.get_canvas()

def render_frame(t):
    canvas.clear(0xFFFFFF)
    canvas.path(tc.vec(0, GROUND_HEIGHT),
                tc.vec(1, GROUND_HEIGHT)).color(0x0).radius(3).finish()

    def circle(x, y, color):
      canvas.circle(tc.vec(x, y)).color(ti.rgb_to_hex(color)).radius(7).finish()

    for i in range(n_springs):
      def get_pt(x):
        return tc.vec(x[0], x[1])

      a = act[t - 1, i] * 0.5
      r = 2
      if spring_actuation[i] == 0:
        a = 0
        c = 0x222222
      else:
        r = 4
        c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
      canvas.path(
          get_pt(x[t, spring_anchor_a[i]]),
          get_pt(x[t, spring_anchor_b[i]])).color(c).radius(r).finish()

    for i in range(n_objects):
      color = (0.4, 0.6, 0.6)
      if i == HEAD_ID:
        color = (0.8, 0.2, 0.3)
      circle(x[t, i][0], x[t, i][1], color)

    gui.update()


### TRAINING SYSTEMS ###
@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = -x[t, HEAD_ID][0]

@ti.kernel
def clear_states():
  for t in range(0, MAX_STEPS):
    for i in range(0, n_objects):
      x.grad[t, i] = ti.Vector([0.0, 0.0])
      v.grad[t, i] = ti.Vector([0.0, 0.0])
      v_inc[t, i] = ti.Vector([0.0, 0.0])
      v_inc.grad[t, i] = ti.Vector([0.0, 0.0])

  for i in range(N_HIDDEN):
    for j in range(n_input_states()):
      weights1.grad[i, j] = 0.0
    bias1.grad[i] = 0.0

  for i in range(n_springs):
    for j in range(N_HIDDEN):
      weights2.grad[i, j] = 0.0
    bias2.grad[i] = 0.0

def forward(output=None, visualize=True, visualize_sleep=None):
  interval = VIS_INTERVAL
  total_steps = SIMULATION_TRAINING_STEPS if not output else MAX_STEPS

  for t in range(1, total_steps):
    # 1. Simulation step
    compute_center_of_mass(t - 1)
    policy_compute_layer1(t - 1)
    policy_compute_layer2(t - 1)
    apply_spring_force(t - 1)

    if USE_TIME_OF_IMPACT:
      advance_toi(t)
    else:
      advance_no_toi(t)

    # 2. Rendering (at a given interval)
    if (t + 1) % interval == 0 and visualize:
      # Limit the frame rate for visualisation purposes
      if visualize_sleep is not None:
        time.sleep(visualize_sleep)

      render_frame(t)

  # 3. Loss to optimize for
  loss[None] = 0
  compute_loss(total_steps - 1)


def train(visualize):
  # Logging
  writer = SummaryWriter()

  # Initialize the policy network
  policy_init_network()

  for iter in range(TRAINING_EPOCHS):
    clear_states()

    # Forward pass and gradient tracking
    with ti.Tape(loss):
      forward(visualize=visualize)

    print('Iter=', iter, 'Loss=', loss[None])
    writer.add_scalar("Loss", loss[None], iter)

    # Compute L2 norm of the weights
    total_norm_sqr = 0
    for i in range(N_HIDDEN):
      for j in range(n_input_states()):
        total_norm_sqr += weights1.grad[i, j]**2
      total_norm_sqr += bias1.grad[i]**2

    for i in range(n_springs):
      for j in range(N_HIDDEN):
        total_norm_sqr += weights2.grad[i, j]**2
      total_norm_sqr += bias2.grad[i]**2

    print(total_norm_sqr)

    # Optimization algorithm step, important trick -- gradient normalization (used in RNNs)
    scale = LEARNING_RATE / (total_norm_sqr**0.5 + 1e-6)
    for i in range(N_HIDDEN):
      for j in range(n_input_states()):
        weights1[i, j] -= scale * weights1.grad[i, j]
      bias1[i] -= scale * bias1.grad[i]

    for i in range(n_springs):
      for j in range(N_HIDDEN):
        weights2[i, j] -= scale * weights2.grad[i, j]
      bias2[i] -= scale * bias2.grad[i]



robot_id = 0
if len(sys.argv) != 3:
  print(
      "Usage: python3 mass_spring.py [robot_id=0, 1, 2, ...] [task=train/plot]")
  exit(-1)
else:
  robot_id = int(sys.argv[1])
  task = sys.argv[2]

def main():
  # Configure the robot of interest
  setup_robot(*robots[robot_id]())

  # Start optimization
  train(visualize=True)

  # Test original problem
  clear_states()
  forward('Robot-{}, Original Stiffness (3e4)'.format(robot_id), visualize=True, visualize_sleep=0.016)


if __name__ == '__main__':
  main()
