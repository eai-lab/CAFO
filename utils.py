import numpy as np
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger


def setup_neptune_logger(cfg: DictConfig, tags: list = None):
    """
    Nettune AI loger configuration. Needs API key.
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :param tags: List of tags to log a particular run. [list]
    :return:
    """
    meta_tags = list(get_values(cfg))

    # setup logger
    neptune_logger = NeptuneLogger(
        api_key=cfg.logger.api_key,
        project=cfg.logger.project_name,
        tags=meta_tags,  # mode="debug"
        mode="debug",
    )

    neptune_logger.experiment["parameters/model"] = cfg.model
    neptune_logger.experiment["parameters/experiment_num"] = cfg.exp_num
    neptune_logger.experiment["parameters/task"] = cfg.task.task_name
    neptune_logger.experiment["parameters/optimizer"] = cfg.task.optimizer
    neptune_logger.experiment["parameters/batch_size"] = cfg.dataset.batch_size
    neptune_logger.experiment["parameters/channelattention_type"] = cfg.channelattention.name
    neptune_logger.experiment["parameters/cv_num"] = cfg.task.validation_cv_num

    neptune_logger.experiment["parameters/run"] = {
        k: v for k, v in cfg.items() if not isinstance(v, omegaconf.dictconfig.DictConfig)
    }

    return neptune_logger


def get_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from get_values(v)
        else:
            yield str(v)


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
