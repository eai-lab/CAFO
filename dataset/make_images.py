from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
import numpy as np


def generate_rp(sensor, cfg, seed=0, meaningless_sensor=None):
    """inputs a sensor array, returns recurrence plots all stacked
    Args:
        sensor: np.array.shape of (160,14)
        cfg: config file
        seed: fix random seed
    Returns:
        stacked_img: np.array.shape of (14, 160, 160)
    """
    rp = RecurrencePlot(
        dimension=cfg.task.recurrenceplot.dimension,
        time_delay=cfg.task.recurrenceplot.time_delay,
        threshold=cfg.task.recurrenceplot.threshold,
        percentage=cfg.task.recurrenceplot.percentage,
    )
    num_cols = sensor.shape[1]
    imgs = []

    column_idx_list = list(range(num_cols))

    # remove feature if given
    if cfg.remove_feature:
        for idx in cfg.remove_feature_idx_lists:
            column_idx_list.remove(idx)

    for i in column_idx_list:
        col_img = rp.fit_transform(sensor[:, i].reshape(1, -1))[0]
        imgs.append(col_img)

    if meaningless_sensor is not None:
        meaningless_img = rp.fit_transform(meaningless_sensor)[0]
        imgs.append(meaningless_img)

    stacked_img = np.dstack(imgs)
    reshaped_img = np.transpose(stacked_img, (2, 0, 1))

    return reshaped_img


def generate_gramian(sensor, cfg, seed=0, meaningless_sensor=None):
    """inputs a sensor array, returns recurrence plots all stacked
    Args:
        sensor: np.array.shape of (160,14)
    Returns:
        stacked_img: np.array.shape of (14, 160, 160)
    """
    gramian = GramianAngularField()
    num_cols = sensor.shape[1]
    imgs = []

    for i in range(num_cols):
        col_img = gramian.fit_transform(sensor[:, i].reshape(1, -1))[0]
        imgs.append(col_img)

    if meaningless_sensor is not None:
        meaningless_img = gramian.fit_transform(meaningless_sensor)[0]
        imgs.append(meaningless_img)

    stacked_img = np.dstack(imgs)
    reshaped_img = np.transpose(stacked_img, (2, 0, 1))

    return reshaped_img


def generate_markov(sensor, cfg, seed=0, meaningless_sensor=None):
    """inputs a sensor array, returns recurrence plots all stacked
    Args:
        sensor: np.array.shape of (160,14)
    Returns:
        stacked_img: np.array.shape of (14, 160, 160)
    """
    mtf = MarkovTransitionField(n_bins=8)
    num_cols = sensor.shape[1]
    imgs = []

    for i in range(num_cols):
        col_img = mtf.fit_transform(sensor[:, i].reshape(1, -1))[0]
        imgs.append(col_img)

    if meaningless_sensor is not None:
        meaningless_img = mtf.fit_transform(meaningless_sensor)[0]
        imgs.append(meaningless_img)

    stacked_img = np.dstack(imgs)
    reshaped_img = np.transpose(stacked_img, (2, 0, 1))

    return reshaped_img
