# 3rd-party module
from torch import nn

def loss_function(stance_predict, stance_target,
                  sentiment_predict, sentiment_target,
                  lexicon_vector,
                  stance_weight, sentiment_weight,
                  stance_loss_weight, lexicon_loss_weight):

    # get cross entropy loss
    ce_loss = nn.CrossEntropyLoss()
    stance_loss = ce_loss(stance_predict, stance_target)
    sentiment_loss = ce_loss(sentiment_predict, sentiment_target)

    # get attention weight
    sum_weight = stance_weight + sentiment_weight
    norm_weight = sum_weight / sum_weight.max(dim=1, keepdim=True)[0]

    # get MSE loss (lexicon loss)
    mse_loss = nn.MSELoss()
    lexicon_loss = mse_loss(norm_weight, lexicon_vector)

    # get final loss
    total_loss = stance_loss_weight * stance_loss + \
                 (1-stance_loss_weight) * sentiment_loss + \
                 lexicon_loss_weight * lexicon_loss

    return total_loss, (stance_loss, sentiment_loss, lexicon_loss)