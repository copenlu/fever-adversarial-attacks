import heapq
from copy import deepcopy
from functools import partial
from operator import itemgetter
from typing import List
from typing import Tuple

import numpy
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

from attack_multiple_objectives.nli_utils import NLI_DIC_LABELS


def get_embedding_weight_bert(model, model_type='bert'):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    if model_type == 'bert':
        return model.bert.embeddings.word_embeddings.weight.cpu().detach()
    elif model_type == 'roberta':
        return model.roberta.embeddings.word_embeddings.weight.cpu().detach()


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


def add_hooks_bert(model, model_type='bert'):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    if model_type == 'bert':
        model.bert.embeddings.word_embeddings.weight.requires_grad = True
        model.bert.embeddings.word_embeddings.register_backward_hook(extract_grad_hook)
    elif model_type == 'roberta':
        model.roberta.embeddings.word_embeddings.weight.requires_grad = True
        model.roberta.embeddings.word_embeddings.register_backward_hook(extract_grad_hook)


def evaluate_batch_bert(model: torch.nn.Module, batch: Tuple, trigger_token_ids: List = None, reduction='mean'):
    # Attach attack_multiple_objectives if present
    input_ids = batch[0]
    loss_f = torch.nn.CrossEntropyLoss(reduction=reduction)
    if trigger_token_ids is not None:
        trig_tensor = torch.cuda.LongTensor(trigger_token_ids)
        trig_tensor = trig_tensor.repeat(batch[0].shape[0], 1)
        # CLS trigger tokens
        input_ids = torch.cat([input_ids[:, 0].unsqueeze(1), trig_tensor, input_ids[:, 1:]], 1)

    loss, logits_val = model(input_ids, attention_mask=input_ids > 1, labels=batch[1])
    loss = loss_f(logits_val, batch[1].long())
    labels = batch[1]

    return loss, logits_val, labels


def evaluate_batch_ppl(model: torch.nn.Module, batch: Tuple, tokenizer, trigger_token_ids: List = None):
    # Attach attack_multiple_objectives if present
    input_ids = batch[0]

    if trigger_token_ids == None:
        trigger_token_ids = []

    input_ids_claim = []
    for instance in input_ids:
        instance = instance.detach().cpu().numpy().tolist()
        # get the claim including SEP token, add triggger and append claim plus evidence
        claim_tokens = instance[0:1] + trigger_token_ids + instance[1:]
        claim_tokens = claim_tokens[:512]
        input_ids_claim.append(claim_tokens)

    # pad batch
    max_len = min(512, max([len(i) for i in input_ids_claim]))
    input_ids_claim = [instance + [tokenizer.pad_token_id] * (max_len - len(instance))
                       for instance in input_ids_claim]
    input_ids = torch.tensor(input_ids_claim).cuda()

    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]
    return loss, prediction_scores


def evaluate_batch_gpt(model: torch.nn.Module, batch: Tuple, trigger_token_ids: List = None, tokenizer=None):
    # Attach attack_multiple_objectives if present
    input_ids = batch[0]
    loss_f = torch.nn.CrossEntropyLoss()
    input_ids_claim = []

    if trigger_token_ids == None:
        trigger_token_ids = []

    # index_sep = (input_ids == tokenizer.sep_token_id).nonzero()[:, 1]
    for instance in input_ids:
        instance = instance.detach().cpu().numpy().tolist()
        # get the claim including SEP token, add triggger and append again claim but without the CLS token
        # CLS, trigger, claim tokens
        sep_index = instance.index(tokenizer.sep_token_id)
        claim_tokens = instance[0:1] + trigger_token_ids + instance[1:]
        input_ids_claim.append(claim_tokens[:512])
        # pad batch
        max_len = max([len(i) for i in input_ids_claim])
        input_ids_claim = [instance + [tokenizer.pad_token_id] * (max_len - len(instance))
                           for instance in input_ids_claim]
        input_ids = torch.tensor(input_ids_claim).cuda()

    # eval is w.r.t. entailment - this is the target class in the NLI case,
    # i.e. the one we want to minimize the loss for.
    logits_val = model(input_ids, attention_mask=input_ids != tokenizer.pad_token_id)[0]
    loss = loss_f(logits_val, batch[1].long())

    return loss, logits_val, batch[1].long()


def evaluate_batch_nli(model: torch.nn.Module, batch: Tuple, trigger_token_ids: List = None, tokenizer=None):
    # Attach attack_multiple_objectives if present
    input_ids = batch[0]
    loss_f = torch.nn.CrossEntropyLoss()
    input_ids_claim = []

    if trigger_token_ids == None:
        trigger_token_ids = []

    # index_sep = (input_ids == tokenizer.sep_token_id).nonzero()[:, 1]
    for instance in input_ids:
        instance = instance.detach().cpu().numpy().tolist()
        sep_index = instance.index(tokenizer.sep_token_id)
        # get the claim including SEP token, add triggger and append again claim but without the CLS token
        claim_tokens = instance[0:1] + trigger_token_ids + instance[1:sep_index + 1] + instance[1:sep_index + 1]
        input_ids_claim.append(claim_tokens[:512])
        # pad batch
        max_len = max([len(i) for i in input_ids_claim])
        input_ids_claim = [instance + [tokenizer.pad_token_id] * (max_len - len(instance))
                           for instance in input_ids_claim]
        input_ids = torch.tensor(input_ids_claim).cuda()

    # eval w.r.t. entailment - this is the target class in the NLI case,
    # i.e. the one we want to minimize the loss for.
    logits_val = model(input_ids, attention_mask=input_ids != tokenizer.pad_token_id)[0]
    loss = loss_f(logits_val, batch[1].long())

    return loss, logits_val, batch[1].long()


def get_average_grad_transformer(model, batch, trigger_token_ids, batch_func, target_label=None, ):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend attack_multiple_objectives to the batch
    original_labels = batch[1].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[1] = int(target_label) * torch.ones_like(batch[1]).cuda()
    global extracted_grads
    extracted_grads = []  # clear existing stored grads
    loss, logits, labels = batch_func(model, batch, trigger_token_ids)
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].detach().cpu()
    batch[1] = original_labels.detach()  # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    # start from position 1 as at 0 is the CLS token
    averaged_grad = averaged_grad[1:len(trigger_token_ids) + 1]  # return just trigger grads
    return averaged_grad


def get_loss_per_candidate(model, model_nli, gpt_model, batch, trigger_token_ids, cand_trigger_token_ids, tokenizer,
                           nli_w=0.0, fc_w=1.0, ppl_w=0.0, idx=0):

    nli_batch_func = partial(evaluate_batch_nli, tokenizer=tokenizer)
    gpt_batch_func = partial(evaluate_batch_gpt, tokenizer=tokenizer)

    original_labels = batch[1].clone()
    loss_per_candidate = get_loss_per_candidate_bert(idx, model, batch, trigger_token_ids,
                                                     cand_trigger_token_ids,
                                                     evaluate_batch_bert, tokenizer)  # uses the real labels
    loss_per_candidate = [(_t, _s * fc_w) for _t, _s in loss_per_candidate]

    if nli_w > 0.0:
        batch[1] = int(NLI_DIC_LABELS['contradiction']) * torch.ones_like(batch[1]).cuda()
        nli_loss_per_candidate = get_loss_per_candidate_bert(idx, model_nli, batch, trigger_token_ids,
                                                             cand_trigger_token_ids,
                                                             nli_batch_func, tokenizer)
        loss_per_candidate = [(_t, _s + nli_loss_per_candidate[i][1] * nli_w)
                              for i, (_t, _s) in enumerate(loss_per_candidate)]
        batch[1] = original_labels.detach()
    if ppl_w > 0.0:
        batch[1] = 0 * torch.ones_like(batch[1]).cuda()
        gpt_loss_per_candidate = get_loss_per_candidate_bert(idx, gpt_model, batch, trigger_token_ids,
                                                             cand_trigger_token_ids,
                                                             gpt_batch_func, tokenizer)
        loss_per_candidate = [(_t, _s + gpt_loss_per_candidate[i][1] * ppl_w)
                              for i, (_t, _s) in enumerate(loss_per_candidate)]
        loss_per_candidate += ppl_w * gpt_loss_per_candidate
        batch[1] = original_labels.detach()

    return loss_per_candidate


def get_best_candidates_all_obj(model, model_nli, gpt_model, batch, trigger_token_ids, cand_trigger_token_ids,
                                tokenizer, beam_size=1, nli_w=0.0, fc_w=1.0, ppl_w=0.0):
    loss_per_candidate = get_loss_per_candidate(model, model_nli, gpt_model, batch,
                                                trigger_token_ids, cand_trigger_token_ids, tokenizer,
                                                nli_w=nli_w, fc_w=fc_w, ppl_w=ppl_w, idx=0)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)):  # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_ = get_loss_per_candidate(model, model_nli, gpt_model, batch,
                                           cand, cand_trigger_token_ids, tokenizer,
                                           nli_w=nli_w, fc_w=fc_w, ppl_w=ppl_w, idx=idx)

            loss_per_candidate.extend(loss_)
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]


def get_best_candidates_bert(model, batch, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate_bert(0, model, batch, trigger_token_ids,
                                                     cand_trigger_token_ids, evaluate_batch_bert, tokenizer)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)):  # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(
                get_loss_per_candidate_bert(idx, model, batch, cand, cand_trigger_token_ids, evaluate_batch_bert, tokenizer))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate_bert(index, model, batch, trigger_token_ids, cand_trigger_token_ids, eval_batch_f, tokenizer):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate attack_multiple_objectives it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss, logits, labels = eval_batch_f(model, batch, trigger_token_ids)
    curr_loss = curr_loss.cpu().detach().numpy()

    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        token = tokenizer.convert_ids_to_tokens([cand_trigger_token_ids[index][cand_id]])
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)  # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id]  # replace one token
        if not any(_s.isalpha() for _s in token):
            loss = -100.0
        elif not (token[0].startswith('Ġ') or token[0].istitle()):
            loss = -100.0
        else:
            loss, logits, labels = eval_batch_f(model, batch, trigger_token_ids_one_replaced)
            loss = loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def eval_fc(model: torch.nn.Module, test_dl: BatchSampler, trigger_token_ids: List = None, labels_num=3):
    model.eval()
    with torch.no_grad():
        labels_all = []
        logits_all = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            # Attach triggers if present
            loss, logits_val, labels = evaluate_batch_bert(model, batch, trigger_token_ids)

            labels_all += labels.detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = numpy.argmax(numpy.asarray(logits_all).reshape(-1, labels_num), axis=-1)
        acc = sum(prediction == labels_all) / len(labels_all)

    return acc


def eval_ppl(model: torch.nn.Module, test_dl: BatchSampler, tokenizer, trigger_token_ids: List = None):
    model.eval()
    with torch.no_grad():
        ppl_loss = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            # Attach triggers if present
            loss, prediction_scores = evaluate_batch_ppl(model, batch, tokenizer, trigger_token_ids)
            loss = torch.exp(loss)
            ppl_loss.append(loss.item() / batch[0].shape[0])
    return numpy.mean(ppl_loss, dtype=float), numpy.std(ppl_loss)


def eval_nli(model: torch.nn.Module, test_dl: BatchSampler, tokenizer, trigger_token_ids: List = None, labels_num=3):
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        logits_all = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            loss, logits_val, _ = evaluate_batch_nli(model, batch, trigger_token_ids, tokenizer)
            logits_all += softmax(logits_val).detach().cpu().numpy().tolist()

        # for each class, get number of instances
        prediction = numpy.argmax(numpy.asarray(logits_all).reshape(-1, labels_num), axis=-1)
        unique, counts = numpy.unique(prediction, return_counts=True)
        pred_dict = dict(zip(unique, counts))
        pred_dict = {int(k): int(v) for k, v in pred_dict.items()}

    return pred_dict
