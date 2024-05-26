import sys
import warnings

from unirec import unirecError
from unirec.commands import create_argument_parser
from unirec.configuration import load_configuration
from unirec.logging import get_logger, enable_verbose_logging

def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        if arguments.command == 'train':
            from unirec.commands.train import entrypoint
        elif arguments.command == 'eval':
            from unirec.commands.eval import entrypoint
        else:
            raise unirecError(
                f'do not support command {arguments.command}')
        params = load_configuration(arguments.configuration)

        params['best_epoch'] = arguments.best_epoch
        entrypoint(params)
    except unirecError as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
