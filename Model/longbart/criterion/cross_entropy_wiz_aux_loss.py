from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion



@register_criterion('label_smoothed_cross_entropy_with_aux_loss')
class LabelSmoothedCrossEntropyCriterionWizAuxLoss(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

                Returns a tuple with three elements:
                1) the loss
                2) the sample size, which is used as the denominator for the gradient
                3) logging outputs to display while training
                """
        net_output = model(**sample['net_input'])
        aux_loss = net_output[1]['aux_loss']
        if type(aux_loss) is not int:
            logging_aux_loss = aux_loss.data
        else:
            logging_aux_loss = aux_loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'aux_loss': logging_aux_loss
        }
        loss += aux_loss
        return loss, sample_size, logging_output