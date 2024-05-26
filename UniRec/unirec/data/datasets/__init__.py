from unirec.data.datasets.dataset import Dataset
from unirec.data.datasets.tempo.uni import unirecDataset
from unirec.data.datasets.tempo.unict import unirecctDataset

_SUPPORTED_DATASETS = {
    'pos': Dataset,
    'tempo_unirec': unirecDataset,
    'tempo_unirecct': unirecctDataset
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    dataloader_type = params['dataset'].get('dataloader', 'pos')
    try:
        return _SUPPORTED_DATASETS[dataloader_type](params).data
    except KeyError:
        raise KeyError(f'Not support {dataloader_type} dataset')
