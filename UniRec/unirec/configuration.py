import json
import os

from unirec import unirecError


def load_configuration(descriptor):

    if not os.path.exists(descriptor):
        raise unirecError(f'Configuration file {descriptor} '
                          f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)
