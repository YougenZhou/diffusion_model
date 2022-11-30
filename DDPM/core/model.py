

class ModelInterface(object):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Model')
        return group

    def __init__(self, args, model_cls):
        model = model_cls(args)
        self.model = model
        self.loss_scaler = None
        self.optimizer = None

    def train_step(self, inputs):
        self.model.train()
        metrics = self.model(inputs)
