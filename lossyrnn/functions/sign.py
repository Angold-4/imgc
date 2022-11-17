from torch.autograd import Function

class Sign(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(Sign, self).__init__()

    @staticmethod
    def forward(_, input, is_training=True):
        # Apply quantization noise while only training ("Probility")
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            # In order to have a fixed representation for a particular input
            # after trained, only the most likely result will be considered
            return input.sign()

    @staticmethod
    def backward(_, grad_output):
        return grad_output, None
