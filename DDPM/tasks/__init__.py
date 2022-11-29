from DDPM.core.task import Task
from DDPM.utils import parse_args


TASK_REGISTER = {}


__all__ = [
    'TASK_REGISTER',
    'register_task',
    'create_task'
]


def register_task(name):

    def __wrapped__(cls):
        if name in TASK_REGISTER:
            raise ValueError(f'Cannot register duplicate task ({name})')
        if not issubclass(cls, Task):
            raise ValueError(f'Task ({name}) must extend Task')
        TASK_REGISTER[name] = cls
        return cls

    return __wrapped__


def create_task(args) -> Task:
    return TASK_REGISTER[args.task](args)


def add_cmdline_args(parser):
    group = parser.add_argument_group('Task')
    group.add_argument('--task', type=str, required=True, choices=list(TASK_REGISTER.keys()))

    args = parse_args(parser, allow_unknown=True)
    if args.task not in TASK_REGISTER:
        raise ValueError(f'Unknown task type: {args.task}')
    TASK_REGISTER[args.task].add_cmdline_args(parser)
    return group


import DDPM.tasks.dialog_generation
