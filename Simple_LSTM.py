# model to predict the 5th number from the sequence of 4 consecutive numbers

import taichi as ti
import numpy as np
import random

real = ti.f32
NUM_INPUT = 1
NUM_HIDDEN = 4  # 4 disconnected hidden cells
NUM_OUTPUT = 1
NUM_CONCAT = 1 + NUM_INPUT  # 1 is number of input to each hidden cell
TIME_STEPS = 4  # corresponds to 4 input of x's components
LR = 0.1
BATCH_SIZE = 100
PRECISION = 3

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

# weights:
wg = scalar()
wi = scalar()
wf = scalar()
wo = scalar()
w_result = scalar()
# states:
g = scalar()
i = scalar()
f = scalar()
o = scalar()
s = scalar()
h = scalar()
x = scalar()
xc = scalar()
# bias:
bg = scalar()
bi = scalar()
bf = scalar()
bo = scalar()
b_result = scalar()
# loss:
loss = scalar()
# output:
y = scalar()
result = scalar()
# temp:
temp_g = scalar()
temp_i = scalar()
temp_f = scalar()
temp_o = scalar()
temp_result = scalar()

# place:
ti.root.dense(ti.ij, (NUM_HIDDEN, NUM_CONCAT)).place(wf, wg, wi, wo)
ti.root.dense(ti.ij, (1, NUM_HIDDEN)).place(w_result)
ti.root.dense(ti.ij, (TIME_STEPS, 1)).place(g, i, f, o, s, h, x)
ti.root.dense(ti.ij, (NUM_HIDDEN, 1)).place(bg, bi, bf, bo)
ti.root.dense(ti.ij, (TIME_STEPS, NUM_CONCAT)).place(xc)
ti.root.dense(ti.ij, (TIME_STEPS, 1)).place(temp_g, temp_i, temp_f, temp_o)
ti.root.place(loss)
ti.root.place(y, result, b_result, temp_result)
ti.root.lazy_grad()

# variable initialization:
wg.from_numpy(np.random.rand(NUM_HIDDEN, NUM_CONCAT))
wi.from_numpy(np.random.rand(NUM_HIDDEN, NUM_CONCAT))
wf.from_numpy(np.random.rand(NUM_HIDDEN, NUM_CONCAT))
wo.from_numpy(np.random.rand(NUM_HIDDEN, NUM_CONCAT))
w_result.from_numpy(np.random.rand(1, NUM_HIDDEN))
bi.from_numpy(np.random.rand(NUM_HIDDEN, 1))
bg.from_numpy(np.random.rand(NUM_HIDDEN, 1))
bf.from_numpy(np.random.rand(NUM_HIDDEN, 1))
bo.from_numpy(np.random.rand(NUM_HIDDEN, 1))

# samples_initialization
input_samples = []
output_samples = []


def generate_sample():
    for m in range(BATCH_SIZE):  # generates (a,x,y,z) for in put and b for output in which 0.1<=a<x<y<b<=0.9
        a = round(random.random(), PRECISION)
        b = round(random.random(), PRECISION)
        while 0.1 > a or a >= 0.9:
            a = round(random.random(), PRECISION)
        while a >= b or b >= 0.9:
            b = round(random.random(), PRECISION)
        input_samples.append(
            np.array([[round(a + j * (b - a) / TIME_STEPS, PRECISION)] for j in range(TIME_STEPS)]))
        output_samples.append(round(b, PRECISION))


@ti.kernel
def concat_input():
    for t in range(TIME_STEPS):
        for m in range(NUM_INPUT):
            xc[t, m] = x[t, m]
    for t in range(TIME_STEPS):
        for m in range(NUM_INPUT, NUM_CONCAT):
            if t > 0:
                xc[t, m] = h[t - 1, m]


@ti.func
def sigmoid(arg: ti.var):
    return 1 / (1 + ti.exp(-arg))


@ti.kernel
def forward_prop():
    for t in range(TIME_STEPS):
        for j in range(NUM_HIDDEN):
            for m in range(NUM_CONCAT):
                temp_g[t, 0] += wg[j, m] * xc[t, m]
                temp_i[t, 0] += wi[j, m] * xc[t, m]
                temp_f[t, 0] += wf[j, m] * xc[t, m]
                temp_o[t, 0] += wo[j, m] * xc[t, m]
    for t in range(TIME_STEPS):
        for j in range(NUM_HIDDEN):
            g[t, j] = ti.tanh(temp_g[t, 0] + bg[j, 0])
            i[t, j] = sigmoid(temp_i[t, 0] + bi[j, 0])
            f[t, j] = sigmoid(temp_f[t, 0] + bf[j, 0])
            o[t, j] = sigmoid(temp_o[t, 0] + bo[j, 0])
            if t == 0:
                s[t, j] = g[t, j] * i[t, j]
            else:
                s[t, j] = g[t, j] * i[t, j] + s[t - 1, j] * f[t, j]
            h[t, j] = s[t, j] * o[t, j]
    for j in range(NUM_HIDDEN):
        temp_result[None] += w_result[0, j] * h[TIME_STEPS - 1, j]
    for _ in range(1):
        result[None] = sigmoid(temp_result[None] + b_result[None])


@ti.kernel
def loss_func():
    loss[None] += ti.pow(result[None] - y, 2)


# @ti.kernel
def update():
    for j in range(NUM_HIDDEN):
        for m in range(NUM_CONCAT):
            wg[j, m] -= LR * wg.grad[j, m]
            wi[j, m] -= LR * wi.grad[j, m]
            wo[j, m] -= LR * wo.grad[j, m]
            wf[j, m] -= LR * wf.grad[j, m]

    for j in range(NUM_HIDDEN):
        bg[NUM_HIDDEN - 1, 0] -= bg.grad[NUM_HIDDEN - 1, 0] * LR
        bi[NUM_HIDDEN - 1, 0] -= bi.grad[NUM_HIDDEN - 1, 0] * LR
        bo[NUM_HIDDEN - 1, 0] -= bo.grad[NUM_HIDDEN - 1, 0] * LR
        bf[NUM_HIDDEN - 1, 0] -= bf.grad[NUM_HIDDEN - 1, 0] * LR
        w_result[0, j] -= w_result.grad[0, j] * LR

    for _ in range(1):
        b_result[None] -= LR * b_result.grad[None]

    # reset weights' grads:
    for j in range(NUM_HIDDEN):
        for m in range(NUM_CONCAT):
            wg.grad[j, m] = 0
            wi.grad[j, m] = 0
            wo.grad[j, m] = 0
            wf.grad[j, m] = 0

    for j in range(NUM_HIDDEN):
        bg.grad[NUM_HIDDEN - 1, 0] = 0
        bi.grad[NUM_HIDDEN - 1, 0] = 0
        bo.grad[NUM_HIDDEN - 1, 0] = 0
        bf.grad[NUM_HIDDEN - 1, 0] = 0
        w_result.grad[0, k] = 0

    for t in range(TIME_STEPS):
        temp_g[t, 0] = 0
        temp_i[t, 0] = 0
        temp_f[t, 0] = 0
        temp_o[t, 0] = 0
        temp_result[None] = 0
    for _ in range(1):
        b_result.grad[None] = 0


def print_data(data, a, b):
    for j in range(a):
        for m in range(b):
            print(data[j, m], end=' ')
        for m in range(1):
            print()


if __name__ == '__main__':
    generate_sample()
    for _ in range(1000):
        # x.from_numpy(np.array([[0.3], [0.35], [0.4], [0.45]]))
        # y[None] = 0.5
        for k in range(BATCH_SIZE):
            x.from_numpy(np.array(input_samples[k]))
            y[None] = output_samples[k]
            with ti.Tape(loss=loss):
                concat_input()
                forward_prop()
                loss_func()
            print("loss in ", k + 1, "_th train: ", loss[None], " ||||| pred_y =", result[None], " ||||| real_y = ",
                  y[None])
            update()

    print("------------------------")
    test = np.array([[0.3], [0.35], [0.4], [0.45]])
    x.from_numpy(test)
    forward_prop()
    y[None] = 0.5
    loss_func()
    print("loss in test cast: ", loss[None])
    print("output: ", result[None])
