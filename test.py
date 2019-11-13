"""
Document made to test implementation of documentation builder sphinx
"""

class Foo:
    def __init__(self, requires_grad):
        r"""A kind of Tensor that is to be considered a module parameter.

        Parameters are :class:`~torch.Tensor` subclasses, that have a
        very special property when used with :class:`Module` s - when they're
        assigned as Module attributes they are automatically added to the list of
        its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
        Assigning a Tensor doesn't have such effect. This is because one might
        want to cache some temporary state, like last hidden state of the RNN, in
        the model. If there was no such class as :class:`Parameter`, these
        temporaries would get registered too.

        Arguments:
            :param requires_grad: (bool, optional): Default: `True`
        """
        pass

    def method1(self):
        r"""
        This is still a test of the documentation
        :return:
        """
        pass