from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric

class MinibatchEvalAccuracy(AccuracyPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at EVAL time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchEvalAccuracy metric.
        """
        super(MinibatchEvalAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="eval"
        )

    def __str__(self):
        return "Top1_Acc_MB-EVAL"