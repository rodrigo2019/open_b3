import os

from torch import nn


def list_files(base_path, valid_exts=None, contains=None):
    for (root_dir, dir_names, filenames) in os.walk(base_path):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if valid_exts is None or ext.endswith(valid_exts):
                file_path = os.path.join(root_dir, filename)
                yield file_path


def decode_optuna_param(parameters):
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[parameters["net_arch"]]
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[
        parameters["activation_fn"]]

    return {
        "n_steps": parameters["n_steps"],
        "batch_size": parameters["batch_size"],
        "gamma": parameters["gamma"],
        "learning_rate": parameters["learning_rate"],
        # "ent_coef": ent_coef,
        "clip_range": parameters["clip_range"],
        "n_epochs": parameters["n_epochs"],
        "gae_lambda": parameters["gae_lambda"],
        "max_grad_norm": parameters["max_grad_norm"],
        "vf_coef": parameters["vf_coef"],
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=False,
        ),
    }
