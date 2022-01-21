import torch.nn as nn

class DeepNetwork (nn.module):
    # TODO: Define the Deep Q Network class here.
    # This class should estimate Q(s, a).
    #
    # Hints: 1- Define the architecture of the class in the init method.
    #        2- Define a method that takes a state as input
    #           an outputs action-values of the state.
    #        3- You may also define methods for saving and loading
    #           network weights.