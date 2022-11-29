from DDPM.tasks import register_task
from DDPM.core.task import Task
from DDPM.data.empatheric_dialogues import EmpatheticDialogues


@register_task('DialogGeneration')
class DialogGeneration(Task):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Task')

        args, _ = parser.parse_known_args()
        EmpatheticDialogues.add_cmdline_args(parser)

