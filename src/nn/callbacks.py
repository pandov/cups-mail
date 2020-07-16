from catalyst.dl import ConfusionMatrixCallback as CatalystConfusionMatrixCallback
from .names import get_class_names

class ConfusionMatrixCallback(CatalystConfusionMatrixCallback):

    def __init__(self, **kwargs):
        kwargs['class_names'] = get_class_names()
        super().__init__(**kwargs)

    def _add_to_stats(self, outputs, targets):
        return super()._add_to_stats(outputs[1], targets[1])
