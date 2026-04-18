import os


def make_save_path(
        logbase,
        dataset,
        exp_name
    ):
    savepath = os.path.join(logbase, dataset, exp_name)
    return savepath

