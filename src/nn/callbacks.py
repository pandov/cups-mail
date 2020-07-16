from catalyst.dl import ConfusionMatrixCallback as CatalystConfusionMatrixCallback

class ConfusionMatrixCallback(CatalystConfusionMatrixCallback):

    def _add_to_stats(self, outputs, targets):
        return super()._add_to_stats(outputs[1], targets[1])
