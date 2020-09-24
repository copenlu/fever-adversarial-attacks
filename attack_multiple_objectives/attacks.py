"""
Implementation based on https://github.com/Eric-Wallace/universal-triggers
Contains different methods for attacking models. In particular, given the
gradients for token embeddings, it computes the optimal token replacements.
"""
import numpy
import torch
import torch.nn.functional as F


def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """
    The "Hotflip" attack described in Equation (2) of the
    Universal Adversarial Attacks paper. This code is
    heavily inspired by
    the nice code of Paul Michel here
    https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples,
    the model's
    token embedding matrix, and the current trigger token IDs. It returns the
    top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient
    and tries to increase
    the loss (decrease the model's probability of the true class). For
    targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(
        torch.LongTensor(trigger_token_ids),
        embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad,
                                                  embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1  # lower versus increase the
        # class probability.
    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix,
                                   num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


def pairwise_dot_product(src_embeds, vocab_embeds, cosine=False):
    """Compute the cosine similarity between each word in the vocab and each
    word in the source
    If `cosine=True` this returns the pairwise cosine similarity"""
    # Normlize vectors for the cosine similarity
    if cosine:
        src_embeds = F.normalize(src_embeds, dim=-1, p=2)
        vocab_embeds = F.normalize(vocab_embeds, dim=-1, p=2)
    # Take the dot product
    dot_product = torch.einsum("bij,kj->bik", (src_embeds, vocab_embeds))
    return dot_product


def pairwise_distance(src_embeds, vocab_embeds, squared=False):
    """Compute the euclidean distance between each word in the vocab and each
    word in the source"""
    # We will compute the squared norm first to avoid having to compute all
    # the directions (which would have space complexity B x T x |V| x d)
    # First compute the squared norm of each word vector
    vocab_sq_norm = vocab_embeds.norm(p=2, dim=-1) ** 2
    src_sq_norm = src_embeds.norm(p=2, dim=-1) ** 2
    # Take the dot product
    dot_product = pairwise_dot_product(src_embeds, vocab_embeds)
    # Reshape for broadcasting
    # 1 x 1 x |V|
    vocab_sq_norm = vocab_sq_norm.unsqueeze(0).unsqueeze(0)
    # B x T x 1
    src_sq_norm = src_sq_norm.unsqueeze(2)
    # Compute squared difference
    sq_norm = vocab_sq_norm + src_sq_norm - 2 * dot_product
    # Either return the squared norm or return the sqrt
    if squared:
        return sq_norm
    else:
        # Relu + epsilon for numerical stability
        sq_norm = F.relu(sq_norm) + 1e-20
        # Take the square root
        return sq_norm.sqrt()


def tailor_simple(averaged_grad, embedding_matrix, increase_loss=False):
    """
    Tailor approximation simplified by computing just the largest gradient
    w.r.t. the target label.
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad,
                                                  embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1  # lower versus increase the
        # class probability.

    gradient_dot_embedding_matrix = F.normalize(gradient_dot_embedding_matrix,
                                                p=2, dim=1)
    return gradient_dot_embedding_matrix


def tailor_first(averaged_grad, embedding_matrix, trigger_token_ids,
                 reverse_loss=False, normalize=False):
    """
    Tailor approximation of the larget gradient compared to the gradient of
    the current token,
    all w.r.t. the target class.
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(
        torch.LongTensor(trigger_token_ids),
        embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    new_embed_dot_grad = torch.einsum("bij,kj->bik",
                                      (averaged_grad, embedding_matrix))
    prev_embed_dot_grad = torch.einsum("bij,bij->bi",
                                       (averaged_grad, trigger_token_embeds))

    if reverse_loss:
        neg_dir_dot_grad = prev_embed_dot_grad.unsqueeze(
            -1) + new_embed_dot_grad
    else:
        neg_dir_dot_grad = prev_embed_dot_grad.unsqueeze(
            -1) - new_embed_dot_grad

    if normalize:
        # Compute the direction norm (= distance word/substitution)
        direction_norm = pairwise_distance(trigger_token_embeds,
                                           embedding_matrix)
        # Renormalize
        neg_dir_dot_grad /= direction_norm

    return neg_dir_dot_grad


def hotflip_attack_all(averaged_grad, embedding_matrix,
                       averaged_grad_nli, embedding_matrix_nli,
                       averaged_grad_ppl, embedding_matrix_ppl,
                       nli_w=0.0, fc_w=1, ppl_w=0.0,
                       num_candidates=1):
    """Optimise the adversarial attacks for all objectives
    that have a weight > 0. This is described in the paper in Equation 2.
    """
    neg_dir_dot_grad = fc_w * tailor_simple(averaged_grad, embedding_matrix,
                                            increase_loss=False)
    if nli_w != 0:
        neg_dir_dot_grad_nli = tailor_simple(averaged_grad_nli,
                                             embedding_matrix_nli,
                                             increase_loss=False)  # decrease
        # loss for entailment
        neg_dir_dot_grad += nli_w * neg_dir_dot_grad_nli
    if ppl_w != 0:
        neg_dir_dot_grad_ppl = tailor_simple(averaged_grad_ppl,
                                             embedding_matrix_ppl,
                                             increase_loss=False)  # decrease
        # loss real example
        neg_dir_dot_grad += ppl_w * neg_dir_dot_grad_ppl

    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(neg_dir_dot_grad, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = neg_dir_dot_grad.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples
    and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [[None] * num_candidates for _ in
                             range(len(trigger_token_ids))]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = numpy.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][
                candidate_number] = rand_token
    return new_trigger_token_ids


# steps in the direction of grad and gets the nearest neighbor vector.
def nearest_neighbor_grad(averaged_grad, embedding_matrix, trigger_token_ids,
                          tree, step_size, increase_loss=False,
                          num_candidates=1):
    """
    Takes a small step in the direction of the averaged_grad and finds the
    nearest
    vector in the embedding matrix using a kd-tree.
    """
    new_trigger_token_ids = [[None] * num_candidates for _ in
                             range(len(trigger_token_ids))]
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    if increase_loss:  # reverse the sign
        step_size *= -1
    for token_pos, trigger_token_id in enumerate(trigger_token_ids):
        # take a step in the direction of the gradient
        trigger_token_embed = \
        torch.nn.functional.embedding(torch.LongTensor([trigger_token_id]),
                                      embedding_matrix).detach().cpu().numpy()[
            0]
        stepped_trigger_token_embed = trigger_token_embed + \
                                      averaged_grad[
                                          token_pos].detach().cpu().numpy() * step_size
        # look in the k-d tree for the nearest embedding
        _, neighbors = tree.query([stepped_trigger_token_embed],
                                  k=num_candidates)
        for candidate_number, neighbor in enumerate(neighbors[0]):
            new_trigger_token_ids[token_pos][candidate_number] = neighbor
    return new_trigger_token_ids
