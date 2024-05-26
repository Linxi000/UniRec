import tensorflow as tf

from unirec.models.model import Model
from unirec.models.core import embedding


class AbsSeqRec(Model):
    """
    Abstract sequential recommendation model
    """
    def __init__(self, sess, params, n_users, n_items):
        super(AbsSeqRec, self).__init__(sess, params,
                                        n_users, n_items)
        self.input_scale = params['model']['params'].get(
            'input_scale', False)
        self.seq = None
        self.new_seq = None


    def build_feedict(self, batch, epoch_num, is_training=True):
        raise NotImplementedError('build_feedict method should be '
                                  'implemented in concrete model')

    def export_embeddings(self):
        raise NotImplementedError('export_embeddings method should be '
                                  'implemented in concrete model')

    def _create_placeholders(self):
        super(AbsSeqRec, self)._create_placeholders()

        self.seq_ids = tf.compat.v1.placeholder(name='seq_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])

        self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])

        self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        self.logger.debug('--> Create embedding tables')
        with tf.compat.v1.variable_scope('embedding_tables',
                                         reuse=reuse):

            self.item_embedding_table, self.org_item_embedding_table = \
                embedding(vocab_size=self.n_items,
                          embedding_dim=self.embedding_dim,
                          zero_pad=True,
                          use_reg=self.use_reg,
                          l2_reg=self.l2_emb,
                          scope='item_embedding_table',
                          reuse=reuse)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self._create_posneg_emb_inference()
        self._create_net_inference(name, reuse)

        self.seq_emb = tf.reshape(
            self.seq,
            [tf.shape(self.seq_ids)[0] * self.seqlen,
             self.embedding_dim])

    def _create_posneg_emb_inference(self):
        self.logger.debug('--> Create POS/NEG inference')

        pos_ids = tf.reshape(self.pos_ids,
                             [tf.shape(self.seq_ids)[0] * self.seqlen])

        neg_ids = tf.reshape(self.neg_ids,
                             [tf.shape(self.seq_ids)[0] * self.seqlen])

        self.pos_emb = tf.nn.embedding_lookup(
            self.item_embedding_table, pos_ids)

        self.neg_emb = tf.nn.embedding_lookup(
            self.item_embedding_table, neg_ids)

        self.istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos_ids, 0)),
                                   [tf.shape(self.seq_ids)[0] * self.seqlen])

    def _create_test_inference(self, name, reuse=None):

        test_item_emb = tf.nn.embedding_lookup(self.item_embedding_table,
                                               self.test_item_ids)

        self.test_logits = tf.matmul(self.seq,
                                     tf.transpose(test_item_emb, perm=[0, 2, 1]))

        self.test_logits = self.test_logits[:, -1, :]

    def _create_net_inference(self, name, reuse=None):
        with tf.compat.v1.variable_scope(f'{name}_net_inference',
                                         reuse=reuse):

            self.seq = tf.nn.embedding_lookup(self.item_embedding_table,
                                              self.seq_ids)

            if self.input_scale is True:
                self.logger.info('Scale input sequence')
                self.seq = self.seq * (self.embedding_dim ** 0.5)
            else:
                self.logger.info('DO NOT scale input')

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        raise NotImplementedError('_create_loss method should be '
                                  'implemented in concrete model')
