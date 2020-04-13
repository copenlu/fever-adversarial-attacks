from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
from typing import List
from typing import Tuple
import numpy as np


def get_embedding_weight_bert(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    return model.bert.embeddings.word_embeddings.weight.cpu().detach()

def get_embedding_weight_rnn_cnn(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    return model.embedding.weight.cpu().detach()

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


def add_hooks_bert(model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    model.bert.embeddings.word_embeddings.weight.requires_grad = True
    model.bert.embeddings.word_embeddings.register_backward_hook(extract_grad_hook)

def add_hooks_rnn_cnn(model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    model.embedding.weight.requires_grad = True
    model.embedding.register_backward_hook(extract_grad_hook)


def evaluate_batch_bert(model: torch.nn.Module, batch: Tuple, trigger_token_ids: List = None):
    # Attach triggers if present
    input_ids = batch[0]
    loss_f = torch.nn.CrossEntropyLoss()
    if trigger_token_ids is not None:
        trig_tensor = torch.cuda.LongTensor(trigger_token_ids)
        trig_tensor = trig_tensor.repeat(batch[0].shape[0], 1)
        input_ids = torch.cat([input_ids[:,0].unsqueeze(1), trig_tensor, input_ids[:,1:]], 1)
        #input_ids = torch.cat([input_ids[:,:1], trig_tensor, input_ids[:,2:]], 1)

    loss, logits_val = model(input_ids, attention_mask=input_ids > 1, labels=batch[1])
    loss = loss_f(logits_val, batch[1].long())
    labels = batch[1]

    return loss, logits_val, labels


def evaluate_batch_rnn_cnn(model: torch.nn.Module, batch: Tuple, trigger_token_ids: List = None):
    # Attach triggers if present
    input_ids = batch[0]
    loss_f = torch.nn.CrossEntropyLoss()
    if trigger_token_ids is not None:
        trig_tensor = torch.cuda.LongTensor(trigger_token_ids)
        trig_tensor = trig_tensor.repeat(batch[0].shape[0], 1)
        input_ids = torch.cat([input_ids[:,0].unsqueeze(1), trig_tensor, input_ids[:,1:]], 1)
        #input_ids = torch.cat([input_ids[:,:1], trig_tensor, input_ids[:,2:]], 1)

    predictions = model(input_ids)
    loss_val = loss_f(predictions.squeeze(), batch[1])
    labels = batch[1]

    return loss_val, predictions, labels


def get_average_grad_bert(model, batch, trigger_token_ids, target_label=None):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[1].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[1] = int(target_label) * torch.ones_like(batch[1]).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss, logits, labels = evaluate_batch_bert(model, batch, trigger_token_ids)
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].detach().cpu()
    batch[1] = original_labels.detach() # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)] # return just trigger grads
    #averaged_grad = averaged_grad[1:len(trigger_token_ids)+1]  # return just trigger grads
    return averaged_grad


def get_average_grad_rnn_cnn(model, batch, trigger_token_ids, target_label=None):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[1].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[1] = int(target_label) * torch.ones_like(batch[1]).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss, logits, labels = evaluate_batch_rnn_cnn(model, batch, trigger_token_ids)
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].detach().cpu()
    batch[1] = original_labels.detach() # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)] # return just trigger grads
    #averaged_grad = averaged_grad[1:len(trigger_token_ids)+1]  # return just trigger grads
    return averaged_grad


def get_best_candidates_bert(model, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate_bert(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate_bert(idx, model, batch, cand,
                                                             cand_trigger_token_ids))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]


def get_best_candidates_rnn_cnn(model, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate_rnn_cnn(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate_rnn_cnn(idx, model, batch, cand,
                                                             cand_trigger_token_ids))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate_bert(index, model, batch, trigger_token_ids, cand_trigger_token_ids):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss, logits, labels = evaluate_batch_bert(model, batch, trigger_token_ids)
    curr_loss = curr_loss.cpu().detach().numpy()

    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        loss, logits, labels = evaluate_batch_bert(model, batch, trigger_token_ids_one_replaced)
        loss = loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def get_loss_per_candidate_rnn_cnn(index, model, batch, trigger_token_ids, cand_trigger_token_ids):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss, logits, labels = evaluate_batch_rnn_cnn(model, batch, trigger_token_ids)
    curr_loss = curr_loss.cpu().detach().numpy()

    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        loss, logits, labels = evaluate_batch_rnn_cnn(model, batch, trigger_token_ids_one_replaced)
        loss = loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def perplexity(logits, targets):
    """
    Calculates the perplexity of a sentence based on the output of a language model
    :param logits: The language model output
    :param targets: The expected output tokens
    :return: The perplexity score (float)
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    return np.exp(loss.cpu().item())
