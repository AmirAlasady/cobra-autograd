import numpy as np
from tensor import Tensor
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None


class Loss:
    """Base loss class with numerical stability safeguards"""
    EPSILON = 1e-12

    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss with automatic broadcasting"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        """
        Args:
            pred: Prediction tensor of shape (batch_size, ...)
            target: Target tensor of shape (batch_size, ...)

        Returns:
            MSE loss tensor with automatic broadcasting
        """
        return (pred - target).square().mean()


class CrossEntropy(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Numerically stable implementation
        max_val = pred.max(axis=-1, keepdims=True)
        stable_exp = (pred - max_val).exp()
        softmax = stable_exp / stable_exp.sum(axis=-1, keepdims=True)  # Now works!
        return -(target * softmax.log()).mean()



class SoftmaxCrossEntropyLoss(Loss):
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> 'Tensor':
        # If target has the same number of dimensions as pred,
        # assume it's one-hot encoded; otherwise, assume it's indices.
        if target.ndim == pred.ndim:
            class_indices = target.argmax(axis=-1).astype(np.int64)
        else:
            class_indices = target.astype(np.int64)

        # Numerical stability: compute log-softmax
        max_pred = pred.max(axis=-1, keepdims=True)
        log_sum_exp = (pred - max_pred).exp().sum(axis=-1, keepdims=True).log()
        log_softmax = pred - max_pred - log_sum_exp

        # Flatten log_softmax from shape (batch, seq_len, vocab_size) to (batch * seq_len, vocab_size)
        log_softmax_flat = log_softmax.reshape(-1, log_softmax.shape[-1])
        
        # Flatten class indices from shape (batch, seq_len) to (batch * seq_len, 1)
        class_indices = class_indices.reshape(-1, 1)

        # Gather the log probabilities for the correct classes along axis 1
        nll_loss = -log_softmax_flat.gather(1, class_indices)
        return nll_loss.mean()


class BCELoss(Loss):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Binary Cross Entropy with proper Tensor operations"""
        ones = Tensor(1.0)
        epsilon = Tensor(1e-7)

        # Clip using Tensor instances
        clipped = pred.clip(epsilon, ones - epsilon)
        return -(target * clipped.log() + (ones - target) * (ones - clipped).log()).mean()


class Accuracy:
    """Flexible accuracy metric supporting multiple formats"""
    def __call__(self, pred: 'Tensor', target: 'Tensor') -> float:
        """
        Args:
            pred: (batch_size, num_classes) logits/probabilities
            target: (batch_size,) class indices or (batch_size, num_classes) one-hot

        Returns:
            Accuracy score between 0 and 1
        """
        # Get prediction classes
        pred_classes = pred.argmax(axis=-1).data

        # Handle different target formats
        if target.ndim > 1 and target.shape[-1] > 1:
            # One-hot encoded targets
            target_classes = target.argmax(axis=-1).data
        else:
            # Class indices
            target_classes = target.data.squeeze()

        # Device-aware comparison
        if isinstance(pred_classes, np.ndarray):
            return np.mean(pred_classes == target_classes)
        else:  # cupy
            return float(cp.mean(pred_classes == target_classes))