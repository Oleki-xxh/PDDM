# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
from torchmetrics import Metric

import numpy as np
class MeanIntersectionOverUnion(Metric):
    def __init__(
        self,
        n_classes: int,
        ignore_first_class: bool = False
    ) -> None:
        super().__init__()

        # note that states are automatically reset to their defaults when
        # calling metric.reset()
        self.add_state(
            'confmat',
            default=torch.zeros((n_classes, n_classes), dtype=torch.int64),
            dist_reduce_fx='sum'
        )

        # determine dtype for bincounting later on
        n_classes_squared = n_classes**2
        if n_classes_squared < 2**(8-1)-1:
            self._dtype = torch.int8
        elif n_classes_squared < 2**(16-1)-1:
            self._dtype = torch.int16
        else:
            # it does not matter in our tests
            self._dtype = torch.int64    # equal to long

        self._n_classes = n_classes
        self._ignore_first_class = ignore_first_class
        self.labeled = 0
        self.correct = 0
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # convert dtype to speed up bincounting
        preds_ = preds.to(self._dtype)
        target_ = target.to(self._dtype)
        # compute confusion matrix
        unique_mapping = (target_.view(-1) * self._n_classes + preds_.view(-1))
        cnts = torch.bincount(unique_mapping,
                              minlength=self._n_classes**2)
        confmat = cnts.reshape(self._n_classes, self._n_classes)

        # update internal confusion matrix
        self.confmat += confmat

        k = (target_ >= 0) & (target_ < self._n_classes)

        self.labeled += torch.sum(k)
        self.correct += torch.sum((preds_[k] == target_[k]))



    def compute(self) -> torch.Tensor:
        tp = torch.diag(self.confmat).float()
        sum_pred = torch.sum(self.confmat, dim=0).float()
        sum_gt = torch.sum(self.confmat, dim=1).float()
        '''
        # ignore first class (void)
        if self._ignore_first_class:
            tp = tp[1:]
            sum_pred = sum_pred[1:]
            sum_gt = sum_gt[1:]
            sum_pred -= self.confmat[0, 1:].float()

        # we do want ignore classes without gt pixels
        mask = sum_gt != 0
        tp = tp[mask]
        sum_pred = sum_pred[mask]
        sum_gt = sum_gt[mask]
        '''
        intersection = tp
        union = sum_pred + sum_gt - tp
        iou = intersection/union
        iou = torch.where(torch.isinf(iou), torch.full_like(iou, 1), iou)
        iou = torch.where(torch.isnan(iou), torch.full_like(iou, 0), iou)
        classAcc = intersection / self.confmat.sum(axis=1)
        mean_pixel_acc = np.nanmean(classAcc)

        pixel_acc = self.correct / self.labeled
        return [pixel_acc, mean_pixel_acc, torch.mean(iou)]
'''
def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc
'''






        
