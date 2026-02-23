
# Based on a paper by Jascha Sohl-Dickstein
# https://arxiv.org/abs/2402.06184
# The boundary of neural network trainability is fractal

# also a blog post: https://sohl-dickstein.github.io/2024/02/12/fractal.html

import math

import torch

from torch.func import vmap, grad

from pyt.lib.spaces import map_space, grid
from pyt.lib.util import msave, save

torch.manual_seed(69420)

dev = torch.device("cuda:0")

t_real = torch.float64


torch.set_float32_matmul_precision('high')


# alphas: mean field neural network parametrization; reference [9] from Sohl-Dickstein paper:
# Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A
# mean field view of the landscape of two-layer neural networks.
# Proceedings of the National Academy of Sciences, 115(33):
# E7665–E7671, 2018.

def persistent():
    nonlin = "tanh"
    network_n = 16 # original paper: 16
    training_steps = 100

    alpha_1 = 1 / network_n

    match nonlin:
        case "tanh": # TODO others
            sigma = torch.tanh
            alpha_0 = math.sqrt(2/network_n)
        case _:
            alpha_0 = math.sqrt(1/network_n)

    def y_pred(x, W_0, W_1):
        return alpha_1 * W_1 @ sigma(alpha_0 * W_0 @ x)

    def calculate_loss(D, W_0, W_1):
        x, y = D
        loss = ((y - y_pred(x, W_0, W_1))**2).mean()
        return (loss, loss)

    run_network = grad(calculate_loss, argnums=(1,2), has_aux=True)


    def train_network(eta_0, eta_1, D, W_0_init, W_1_init, training_steps):
        W_0 = W_0_init
        W_1 = W_1_init

        loss_init = calculate_loss(D, W_0, W_1)[0].clamp(min=1e-8)

        loss_record = torch.zeros((), device=W_0.device)

        for index in range(training_steps):
            ((grad_W_0, grad_W_1), loss) = run_network(D, W_0, W_1)
            W_0 = W_0 - grad_W_0 * eta_0
            W_1 = W_1 - grad_W_1 * eta_1
            if index >= training_steps - 20:
                loss_record = loss_record + loss / loss_init

        return (loss_record / 20)

    train_many = vmap(train_network, in_dims=(0, 0, None, 0, 0, None))

    train_many = torch.compile(train_many)

#span = 250, 250
#origin = (span[0] // 2) - 10, (span[1] // 2) - 10

#span = 1e7, 1e7
#origin = 0, 0

span = 5.0 / 3, 5.0 / 3
origin = 2.0, 2.0

stretch = 1, 1

zooms = []

scale = 4096

save_partials = False

# TODO save losses directly as safetensors

def render_fractal(seed):
    torch.manual_seed(seed)

    # always generate random data at fp64, then convert, so that rng is at least somewhat consistent
    # across dtypes

    dataset_x = torch.randn([network_n, dataset_size], dtype=torch.float64, device=dev).to(t_real)
    dataset_y = torch.randn((dataset_size,), dtype=torch.float64, device=dev).to(t_real)

    D = (dataset_x, dataset_y)

    _W_0 = torch.randn([network_n, network_n], dtype=torch.float64, device=dev).to(t_real)
    _W_1 = torch.randn([1, network_n], dtype=torch.float64, device=dev).to(t_real)

    mapping = map_space(origin, span, zooms, stretch, scale)
    (_, (height,width)) = mapping

    canvas = torch.zeros([height, width], dtype=t_real, device=dev)

    # eta = learning rate
    etas = grid(mapping).to(dev)
    etas = torch.pow(10.0, etas)

    eta_0 = etas[:,:,1]
    eta_1 = etas[:,:,0].flip(0)

    cols_per_chunk = 16

    convergence_threshold = 1.0

    last_report = 0

    for col_start in range(0, width, cols_per_chunk):
        col_end = col_start + cols_per_chunk
        e0 = eta_0[:, col_start:col_end].reshape(-1)
        e1 = eta_1[:, col_start:col_end].reshape(-1)

        _W_0_batch = _W_0.unsqueeze(0).expand(height * cols_per_chunk, -1, -1).contiguous()
        _W_1_batch = _W_1.unsqueeze(0).expand(height * cols_per_chunk, -1, -1).contiguous()

        res = train_many(e0, e1, D, _W_0_batch, _W_1_batch, training_steps)
        res = res.nan_to_num(nan=1e6, posinf=1e6, neginf=-1e6)

        canvas[:, col_start:col_end] = res.reshape(height, cols_per_chunk)

        if save_partials and col_end > last_report + 64:
            last_report = col_end
            c = canvas[:, 0:col_end].clone()

            conv = c < convergence_threshold

            t_conv = 1 - c / convergence_threshold
            t_div = torch.log1p(c - convergence_threshold) / torch.log1p(torch.tensor(1e6))

            t_conv = (t_conv * conv)
            t_div = (t_div * ~conv)

            t_conv /= t_conv.max().clamp(min=1e-6)
            t_div /= t_div.max().clamp(min=1e-6)

            msave(t_conv, f"{run_dir}/{training_steps:05d}_conv_{col_start}")
            msave(t_div, f"{run_dir}/{training_steps:05d}_div_{col_start}")


    c = canvas.clone()

    '''
    c = torch.log1p(c) / torch.log1p(torch.tensor(1e6))
    c /= c.max().clamp(min=1e-6)
    msave(c, f"{run_dir}/{training_steps:05d}")
    '''

    conv = c < convergence_threshold

    t_conv = 1 - c / convergence_threshold
    t_div = torch.log1p(c - convergence_threshold) / torch.log1p(torch.tensor(1e6))

    t_conv = (t_conv * conv)
    t_div = (t_div * ~conv)

    t_conv /= t_conv.max().clamp(min=1e-6)
    t_div /= t_div.max().clamp(min=1e-6)

    #msave(t_conv, f"{run_dir}/{training_steps:05d}_conv_final")
    #msave(t_div, f"{run_dir}/{training_steps:05d}_div_final")
    save(torch.stack((t_div, t_conv*0.8, t_conv)).to(dtype=torch.float), f"{run_dir}/{seed:06d}")


def main():

    dataset_size = network_n * (network_n + 1)

    schedule(render_fractal, range(1))



def final():
    # todo colorize
    pass

