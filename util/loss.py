# 3rd-party module
import torch.nn as nn
import torch.nn.functional as F

def loss_function(stance_predict, stance_target,
                  sentiment_predict, sentiment_target,
                  lexicon_vector,
                  stance_weight, sentiment_weight,
                  stance_loss_weight, lexicon_loss_weight,
                  ignore_label):

    # # get cross entropy loss
    stance_loss = F.cross_entropy(stance_predict, stance_target, ignore_index=2)
    sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target, ignore_index=2)

    # stance_loss = F.cross_entropy(stance_predict, stance_target)
    # sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target)

    # get attention weight
    sum_weight = stance_weight + sentiment_weight
    norm_weight = sum_weight / sum_weight.max(dim=1, keepdim=True)[0]

    # get MSE loss (lexicon loss)
    lexicon_loss = F.mse_loss(norm_weight, lexicon_vector)

    # get final loss
    total_loss = (
        # stance_loss + 0
        # + sentiment_loss
        stance_loss_weight * stance_loss
        + (1-stance_loss_weight) * sentiment_loss
        + lexicon_loss_weight * lexicon_loss
    )

    return total_loss, (stance_loss, sentiment_loss, lexicon_loss)