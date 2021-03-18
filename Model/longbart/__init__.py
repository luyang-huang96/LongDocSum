import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from . import longbartmodel, sparse_trans_task
from .criterion.cross_entropy_wiz_aux_loss import LabelSmoothedCrossEntropyCriterionWizAuxLoss