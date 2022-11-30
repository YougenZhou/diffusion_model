

class ModelInterface(object):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Model')
        return group

    def __init__(self, args, model_cls):
        model = model_cls(args)
