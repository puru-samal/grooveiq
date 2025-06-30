import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # Get the number of sequences (batch size) and sequence length from input shape
    # Shape is (N, T, ...) where N is batch size and T is sequence length
    N, T = padded_input.shape[:2]

    # Create position indices tensor [0, 1, 2, ..., T-1] on same device as input
    # Shape: (T,)
    positions_1d = torch.arange(T, device=padded_input.device)

    # Add batch dimension and expand to match batch size
    # [None, :] adds dimension to get shape (1, T)
    # expand(N, -1) repeats to get shape (N, T) 
    positions = positions_1d[None, :].expand(N, -1)

    # Create padding mask by comparing positions with sequence lengths
    # positions has shape (N, T)
    # input_lengths[:, None] has shape (N, 1) for broadcasting
    # True indicates padding positions (where position >= length)
    # False indicates valid positions (where position < length)
    mask = positions >= input_lengths[:, None]

    # Return boolean mask of shape (N, T)
    # True = padding position that should be masked
    # False = valid position that should not be masked
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # Get sequence length from input shape
    T = padded_input.shape[1]
    
    # Create a T x T matrix of ones
    # This will be used as the base for creating the lower triangular matrix
    base_matrix = torch.ones(T, T, device=padded_input.device)
    
    # Create lower triangular matrix using tril
    # True values will be in lower triangle (including diagonal)
    # This represents positions each token can attend to
    causal_mask = torch.tril(base_matrix).bool()
    
    # Invert the mask so True values represent positions to mask out
    # i.e. positions each token should NOT attend to
    # This matches the convention where True = "don't attend"
    attn_mask = ~causal_mask
    
    return attn_mask

