"""Explicit length scaling retains the completed 4-chunk recipe's loss scale."""


def backward_loss_scale(reference_chunks, actual_chunks):
    if actual_chunks < 1 or reference_chunks < 0:
        raise ValueError('invalid chunk count for loss scaling')
    return float(reference_chunks) / actual_chunks if reference_chunks else 1.
