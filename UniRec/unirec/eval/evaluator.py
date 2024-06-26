import numpy as np
from tqdm import tqdm

from unirec.logging import get_logger


class Evaluator:
    @classmethod
    def eval(cls, dataloader, model, item_pops=None, **kwargs):
        """
        Get score on valid/test dataset
        :param dataloader:
        :param model:
        :param item_pops:
        :return:
        """
        ndcg = 0.0
        hr=0.0
        mrr = 0.0 
        n_users = 0
        n_batches = dataloader.get_num_batches()
        bad = 0
        pop = 0.0
        for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
            batch_data = dataloader.next_batch()
            feed_dict = model.build_feedict(batch_data, is_training=False)
            predictions = -model.predict(feed_dict)

            test_item_ids = batch_data[2]
            item_type=batch_data[20]

            for i, pred in enumerate(predictions):
                if np.all(pred == pred[0]):
                    bad += 1
                n_users += 1
                rank = pred.argsort().argsort()[0]

                if rank < 10:
                    ndcg += 1 / np.log2(rank + 2)
                    hr += 1
                    mrr += 1 / (rank + 1)

                if item_pops is not None:
                    indices = pred.argsort()[:10]
                    top_item_ids = [test_item_ids[i][idx] for idx in indices]
                    top_item_pops = np.mean([item_pops[iid] for iid in top_item_ids])
                    pop += top_item_pops

        out = ndcg / n_users, hr / n_users,  mrr / n_users

        if item_pops is not None:
            out += (pop / hr,)
        logger = get_logger()
        logger.info(f'Number of bad results: {bad}')
        return out
