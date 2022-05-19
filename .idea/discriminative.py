from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """Calculate mean embedding

       pred: (bs, height * width, n_filters) - (N, 256*256, 32)
       gt: (bs, height * width, n_instances) - (N, 256*256, 20)"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # (bs, n_loc, n_instances, n_filters) - (N, 256*256, 20, 32)
    gt_expanded = gt.unsqueeze(3)  # (bs, n_loc, n_instances, 1) - (N, 256*256, 20, 1) : 0 (b.g) or 1 (instance area)

    # extract pixel values from each embedding space which belong to leaf instance area
    pred_masked = pred_repeated * gt_expanded  # (N, 256*256, 20, 32)

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # (n_loc, n_objects, n_filters)
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]  # (256*256, n_objects, 32)
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]  # (256*256, n_objects, 1)

        # calculate mean embedding for each instance
        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # (n_objects, 32)
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)  # (20, 32)
        means.append(_mean_sample)

    means = torch.stack(means)  # (N, 20, 32)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    """An intra-cluster pull-force that draws embeddings towards the mean embedding

       pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()  # (N, 256*256, 32)
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                        delta_v, min=0.0) ** 2) * gt[:, :, :, 0]  # (N, 256*256, 20)

    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
        _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """An inter-cluster push-force that pushes clusters away from each other,
    increasing the distance btw the cluster centers

    means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """A small pull force that draws all clusters towards the origin

    means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term


def discriminative_loss(input, target, n_objects,
                        max_n_objects, delta_v, delta_d, norm, usegpu):
    """input: (N, embedding_dims, h, w) - instance segmentation prediction (N, 32, 256, 256)
       target: (N, n_instances, h, w) - GT (N, 20, 256, 256), 20=MAX_N_OBJECTS
                * 20 are not actual # of instances, if actual # of instances are smaller than 20, remainders are filled with zeros
       n_objects: bs - # of clusters(instances) of GT = actual # of instances
       max_n_objects - pre-defined at settings/data_setting.py"""

    alpha = beta = 1.0
    gamma = 0.001

    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)  # =max_n_objects

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)  # (N, 256*256, 32)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)  # (N, 256*256, 20)

    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)

    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v, norm)
    dist_term = calculate_distance_term(
        cluster_means, n_objects, delta_d, norm, usegpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm)

    loss = alpha * var_term + beta * dist_term + gamma * reg_term

    return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm,
                 size_average=True, reduce=True, usegpu=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce
        self.size_average=True  # https://github.com/ashafaei/OD-test/issues/2

        assert self.size_average
        assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        target.requires_grad = False
        return discriminative_loss(input, target, n_objects, max_n_objects,
                                   self.delta_var, self.delta_dist, self.norm,
                                   self.usegpu)
