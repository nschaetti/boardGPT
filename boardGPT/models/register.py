


class ActivationRecorder:
    """
    Records activations
    """

    def __init__(self):
        """
        Constructor
        """
        self.records = {}
    # end def __init__

    @property
    def keys(self):
        """
        Keys
        """
        return list(self.records.keys())
    # end keys

    def clear(self):
        """
        Clears records
        :return:
        """
        self.records = {}
    # end def clear

    def save(
            self,
            name,
            tensor
    ):
        """
        Save a tensor

        Args:
            name: name of the tensor
            tensor: tensor
        """
        # detach pour éviter d'accumuler des gradients
        self.records[name] = tensor.detach().cpu()
    # end def save

    def get(
            self,
            name
    ):
        """
        Get a tensor by name.

        Args:
            name: name of the tensor

        Returns:
            tensor: tensor
        """
        return self.records.get(name, None)
    # end def get

    def all(self):
        """
        Get all tensors
        """
        return self.records
    # end def all

# end class ActivationRecorder
