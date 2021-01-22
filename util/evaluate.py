# 3rd-party module
import torch
import pandas as pd

# self-made module
from util import loss
from util import scorer

def evaluate_function(device, model, config, batch_iterator):
    total_loss = 0.0
    stance_loss, sentiment_loss, lexicon_loss = 0.0, 0.0, 0.0

    all_target = []
    all_stance_label, all_sentiment_label = [], []
    all_stance_pred, all_sentiment_pred = [], []

    # evaluate model
    model.eval()
    with torch.no_grad():
        for target_name, target, claim, lexicon, \
            stance, sentiment in batch_iterator:
            # specify device for data
            target = target.to(device)
            claim = claim.to(device)
            lexicon = lexicon.to(device)
            stance = stance.to(device)
            sentiment = sentiment.to(device)

            # get predict label and attention weight
            stance_pred, sentiment_pred, stance_weight, sentiment_weight = \
                model(target, claim)

            # calculate loss
            batch_loss, \
                (batch_stance_loss, batch_sentiment_loss, batch_lexicon_loss)= \
                loss.loss_function(stance_predict=stance_pred,
                                   stance_target=stance,
                                   sentiment_predict=sentiment_pred,
                                   sentiment_target=sentiment,
                                   lexicon_vector=lexicon,
                                   stance_weight=stance_weight,
                                   sentiment_weight=sentiment_weight,
                                   stance_loss_weight=config.stance_loss_weight,
                                   lexicon_loss_weight=config.lexicon_loss_weight)

            # sum the batch loss
            total_loss += batch_loss
            stance_loss += batch_stance_loss
            sentiment_loss += batch_sentiment_loss
            lexicon_loss += batch_lexicon_loss

            # get target, label and predict
            all_target.extend(target_name)
            all_stance_label.extend(stance.tolist())
            all_sentiment_label.extend(sentiment.tolist())
            all_stance_pred.extend(
                torch.argmax(stance_pred, axis=1).cpu().tolist())
            all_sentiment_pred.extend(
                torch.argmax(sentiment_pred, axis=1).cpu().tolist())

    # evaluate loss
    total_loss /= len(batch_iterator)
    stance_loss /= len(batch_iterator)
    sentiment_loss /= len(batch_iterator)
    lexicon_loss /= len(batch_iterator)

    # evaluate f1 score
    target_f1, macro_f1, micro_f1 = \
        scorer.stance_score(targets=pd.Series(all_target),
                            label_y=all_stance_label,
                            pred_y=all_stance_pred)
    sentiment_f1 = scorer.sentiment_score(label_y=all_sentiment_label,
                                          pred_y=all_sentiment_pred)

    return (total_loss.item(), stance_loss.item(),
            sentiment_loss.item(), lexicon_loss.item(),
            target_f1.item(), macro_f1.item(),
            micro_f1.item(), sentiment_f1.item())