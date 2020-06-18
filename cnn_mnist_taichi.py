import mnist
import taichi as ti
import numpy as np

# Initialization
ti.init(arch=ti.cpu, default_fp=ti.f32)  # debug=True

# Data type shortcuts
real = ti.f32
scalar = lambda: ti.var(dt=real)

# Number of recognized digits
n_numbers = 10

# Image size
n_pixels = 28 ** 2
image_size = 28
WIDTH = 28
HEIGHT = 28
DEPTH = 1

# POLICY HYPERPARAMETERS
N_HIDDEN = 500
# ConvLayer
N_FILTERS = 1
FILTER_SIZE = 3  # receptive field
STRIDE = 1
ZERO_PADDING = 1
N_BIASES = 1
# PoolLayer
P_FILTER_SIZE = 2
P_STRIDE = 2

# Input image size after zero-padding
w_input = WIDTH + 2 * ZERO_PADDING  # 30
h_input = HEIGHT + 2 * ZERO_PADDING  # 30

# Size of the output1 (after ConvLayer)
w_output1 = int((WIDTH - FILTER_SIZE + 2 * ZERO_PADDING) / STRIDE + 1)  # 28
h_output1 = int((HEIGHT - FILTER_SIZE + 2 * ZERO_PADDING) / STRIDE + 1)  # 28
d_output1 = N_FILTERS  # 1
size_output1_conv = w_output1 * h_output1 * d_output1  # 28**2

# Size of the output2 (after PoolLayer)
w_output2 = int((w_output1 - P_FILTER_SIZE) / P_STRIDE + 1)  # 14
h_output2 = int((h_output1 - P_FILTER_SIZE) / P_STRIDE + 1)  # 14
d_output2 = d_output1  # 1
size_output2_pool = w_output2 * h_output2 * d_output2  # 14**2

# TRAINING HYPERPARAMETERS
TRAINING_EPOCHS = 5

# Data types
pixels = scalar()
input_img = scalar()
weights1_conv = scalar()
bias1_conv = scalar()
output1_conv = scalar()
output1_relu = scalar()
output2_pool = scalar()
weights2_pool_fc = scalar()
output3_fc = scalar()
output3_relu = scalar()
weights3_fc_fc = scalar()
output4_fc = scalar()
#output_end = scalar()
needed = scalar()
output_sum = scalar()
loss = scalar()
learning_rate = scalar()


# Data layout configuration (for faster computation)
@ti.layout
def place():
    ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels)
    ti.root.dense(ti.ij, (w_input, h_input)).place(input_img)
    ti.root.dense(ti.ijk, (FILTER_SIZE, FILTER_SIZE, N_FILTERS)).place(weights1_conv)
    ti.root.dense(ti.i, N_FILTERS).place(bias1_conv)
    ti.root.dense(ti.ijk, (w_output1, h_output1, d_output1)).place(output1_conv)
    ti.root.dense(ti.ijk, (w_output1, h_output1, d_output1)).place(output1_relu)
    ti.root.dense(ti.ijk, (w_output2, h_output2, d_output2)).place(output2_pool)
    ti.root.dense(ti.ijkl, (w_output2, h_output2, d_output2, N_HIDDEN)).place(weights2_pool_fc)
    ti.root.dense(ti.i, N_HIDDEN).place(output3_fc)
    ti.root.dense(ti.i, N_HIDDEN).place(output3_relu)
    ti.root.dense(ti.ij, (N_HIDDEN, n_numbers)).place(weights3_fc_fc)
    ti.root.dense(ti.i, n_numbers).place(output4_fc)
    #ti.root.dense(ti.i, n_numbers).place(output_end)
    ti.root.dense(ti.i, n_numbers).place(needed)
    ti.root.place(output_sum)
    ti.root.place(loss)
    ti.root.place(learning_rate)

    # Add gradient variables
    ti.root.lazy_grad()


# Input - Conv - Relu - Pool - FC - Relu - FC - Output


# Zero-padding
@ti.kernel
def zero_p():
    for i in range(w_input):
        for j in range(h_input):
            input_img[i, j] = 0

    for i in range(WIDTH):
        for j in range(HEIGHT):
            input_img[i + 1, j + 1] = pixels[i, j]


# Init network
@ti.kernel
def init_net():
    for k in range(N_FILTERS):
        for i in range(FILTER_SIZE):
            for j in range(FILTER_SIZE):
                weights1_conv[i, j, k] = ti.random() * 0.05

    for i in range(N_FILTERS):
        bias1_conv[i] = 0

    for n in range(d_output2):
        for i in range(w_output2):
            for j in range(h_output2):
                for k in range(N_HIDDEN):
                    weights2_pool_fc[i, j, n, k] = ti.random() * 0.05

    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights3_fc_fc[i, j] = ti.random() * 0.05


# Clear weights
@ti.kernel
def clear_weights_biases():
    for k in range(N_FILTERS):
        for i in range(FILTER_SIZE):
            for j in range(FILTER_SIZE):
                weights1_conv.grad[i, j, k] = 0

    for i in range(N_FILTERS):
        bias1_conv.grad[i] = 0

    for n in range(d_output2):
        for i in range(w_output2):
            for j in range(h_output2):
                for k in range(N_HIDDEN):
                    weights2_pool_fc.grad[i, j, n, k] = 0

    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights3_fc_fc.grad[i, j] = 0


# Clear outputs
@ti.kernel
def clear_outputs():
    for k in range(d_output1):
        for i in range(w_output1):
            for j in range(h_output1):
                output1_conv[i, j, k] = 0
                output1_relu[i, j, k] = 0
                output1_conv.grad[i, j, k] = 0
                output1_relu.grad[i, j, k] = 0

    for k in range(d_output2):
        for i in range(w_output2):
            for j in range(h_output2):
                output2_pool[i, j, k] = 0
                output2_pool.grad[i, j, k] = 0

    for i in range(N_HIDDEN):
        output3_fc[i] = 0
        output3_relu[i] = 0
        output3_fc.grad[i] = 0
        output3_relu.grad[i] = 0

    for i in range(n_numbers):
        output4_fc[i] = 0
        output4_fc.grad[i] = 0


# Conv
@ti.kernel
def conv_layer():
    for n in range(N_FILTERS):
        for i in range(WIDTH - FILTER_SIZE + 1):
            for j in range(HEIGHT - FILTER_SIZE + 1):
                for k in range(FILTER_SIZE):
                    for r in range(FILTER_SIZE):
                        output1_conv[i, j, n] += input_img[i + k, j + r] * weights1_conv[k, r, n]

    for k in range(d_output1):
        for i in range(w_output1):
            for j in range(h_output1):
                output1_conv[i, j, k] += bias1_conv[k]


# Relu
@ti.kernel
def conv_layer_relu():
    for k in range(d_output1):
        for i in range(w_output1):
            for j in range(h_output1):
                output1_relu[i, j, k] = ti.max(0, output1_conv[i, j, k])


# Pool
@ti.kernel
def pool_layer():
    for n in range(d_output1):
        for i in range(0, (w_output1 - 1) // 2):
            for j in range(0, (h_output1 - 1) // 2):
                i1 = 2 * i
                j1 = 2 * j
                output2_pool[i, j, n] = ti.max(ti.max(output1_relu[i1, j1, n], output1_relu[i1 + 1, j1, n]),
                                               ti.max(output1_relu[i1, j1 + 1, n],
                                                      output1_relu[i1 + 1, j1 + 1, n]))


# FC
@ti.kernel
def fc_layer1():
    for n in range(d_output2):
        for i in range(w_output2):
            for j in range(h_output2):
                for k in range(N_HIDDEN):
                    output3_fc[k] += output2_pool[i, j, n] * weights2_pool_fc[i, j, n, k]


# Relu
@ti.kernel
def fc_layer1_relu():
    for i in range(N_HIDDEN):
        output3_relu[i] = ti.max(1e-6, output3_fc[i])


# FC
@ti.kernel
def fc_layer2():
    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            output4_fc[j] += output3_relu[i] * weights3_fc_fc[i, j]


# Compute loss (cross-entropy)
@ti.kernel
def compute_loss():
    for i in range(n_numbers):
        loss[None] += (-needed[i]) * ti.log(output4_fc[i]) + (needed[i] - 1) * ti.log(1 - output4_fc[i])


# Gradient descent
@ti.kernel
def gd():
    for k in range(N_FILTERS):
        for i in range(FILTER_SIZE):
            for j in range(FILTER_SIZE):
                weights1_conv[i, j, k] -= learning_rate * weights1_conv.grad[i, j, k]

    for i in range(N_FILTERS):
        bias1_conv[i] -= learning_rate * bias1_conv.grad[i]

    for n in range(d_output2):
        for i in range(w_output2):
            for j in range(h_output2):
                for k in range(N_HIDDEN):
                    weights2_pool_fc[i, j, n, k] -= learning_rate * weights2_pool_fc.grad[i, j, n, k]

    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights3_fc_fc[i, j] -= learning_rate * weights3_fc_fc.grad[i, j]


def forward():
    conv_layer()
    conv_layer_relu()
    pool_layer()
    fc_layer1()
    fc_layer1_relu()
    fc_layer2()
    compute_loss()


def backward_grad():
    compute_loss.grad()
    fc_layer2.grad()
    fc_layer1_relu.grad()
    fc_layer1.grad()
    pool_layer.grad()
    conv_layer_relu.grad()
    conv_layer.grad()


# MNIST images
training_images = mnist.train_images()
training_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Initialize network
init_net()


# Compute accuracy of predictions on tests
def test_accuracy():
    n_test = len(test_images) // 10
    accuracy = 0
    for i in range(n_test):
        # Input
        curr_image = test_images[i]
        for j in range(WIDTH):
            for k in range(HEIGHT):
                pixels[j, k] = curr_image[j][k] / 255
        for j in range(n_numbers):
            needed[j] = int(test_labels[i] == j)

        clear_weights_biases()
        clear_outputs()
        loss[None] = 0

        forward()

        outputs = []
        for j in range(n_numbers):
            outputs.append(output4_fc[j])
        prediction = outputs.index(max(outputs))  # Digit with higher prediction
        accuracy += int(prediction == test_labels[i])

    return accuracy / n_test


# Training
def main():
    # Training
    losses = []
    accuracies = []
    for n in range(TRAINING_EPOCHS):
        for i in range(len(training_images)):
            learning_rate[None] = 0.01  #5e-3 * (0.1 ** (2 * i // 60000))

            # Input
            curr_image = training_images[i]
            for j in range(WIDTH):
                for k in range(HEIGHT):
                    pixels[j, k] = curr_image[j][k] / 255
            for j in range(n_numbers):
                needed[j] = int(training_labels[i] == j)

            clear_weights_biases()
            clear_outputs()
            loss[None] = 0

            forward()

            curr_loss = loss[None]
            losses.append(curr_loss)
            losses = losses[-100:]
            if i % 100 == 0:
                print('i =', i, ' loss : ', sum(losses) / len(losses))
            if i % 1000 == 0:
                curr_acc = test_accuracy()
                print('test accuracy: {:.2f}%'.format(100 * curr_acc))
                accuracies.append(curr_acc)

            loss.grad[None] = 1

            backward_grad()

            gd()


if __name__ == '__main__':
    main()
