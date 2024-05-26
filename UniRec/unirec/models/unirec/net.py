import numpy as np
import tensorflow as tf

from unirec.models.core import normalize, embedding, feedforward, add_weight, linear_layer
from unirec.models.core.fish import \
    multi_head_attention_blocks as admix_multi_head_attention_blocks
from unirec.models.abs.sas import AbsSasRec
from unirec.models.unirec.Lin_att import multi_head_attention_blocks

class DoubleBranchunirec(AbsSasRec):
    def __init__(self, sess, params, n_users, n_items):
        super(DoubleBranchunirec, self).__init__(sess, params,
                                                 n_users, n_items)
        self.tempo_embedding_dim = params.get('tempo_embedding_dim', 8)
        fism_params = params['model']['params']['fism']
        self.n_fism_elems = fism_params.get('n_items', 50)
        self.beta = fism_params.get('beta', 1.0)
        self.expand_dim = 3
        self.use_year = params['model']['params'].get('use_year', True)
        self.num_contexts = 7
        self.tempo_linspace = params.get('tempo_linspace', 8)
        self.lambda_trans_seq = params['model']['params'].get('lambda_trans_seq', 0.1)
        self.lambda_glob = params['model']['params'].get('lambda_glob', 0.1)
        self.lambda_ctx = params['model']['params'].get('lambda_ctx', 0.1)
        self.residual_type = params['model']['params'].get(
            'residual', 'add')
        self.num_global_heads = 2
        self.num_trans_global_heads = 2
        self.dim_head = self.embedding_dim
        self.ctx_activation = self._activation(
            params['model']['params'].get('ctx_activation', 'none'))
        self.local_output_dim = 2 * self.embedding_dim
        self.lambda_user = params['model']['params'].get('lambda_user', 0.0)
        self.lambda_item = params['model']['params'].get('lambda_item', 0.0)


    def build_feedict(self, batch, epoch_num=0, is_training=True):
        feed_dict = {
            self.user_ids: batch[0],
            self.seq_ids: batch[1],
            self.is_training: is_training
        }
        if is_training is True:
            feed_dict[self.pos_ids] = batch[2]
            feed_dict[self.neg_ids] = batch[3]
            feed_dict[self.seq_year_ids] = batch[4]
            feed_dict[self.seq_month_ids] = batch[5]
            feed_dict[self.seq_day_ids] = batch[6]
            feed_dict[self.seq_dayofweek_ids] = batch[7]
            feed_dict[self.seq_dayofyear_ids] = batch[8]
            feed_dict[self.seq_week_ids] = batch[9]
            feed_dict[self.seq_hour_ids] = batch[10]
            feed_dict[self.pos_year_ids] = batch[11]
            feed_dict[self.pos_month_ids] = batch[12]
            feed_dict[self.pos_day_ids] = batch[13]
            feed_dict[self.pos_dayofweek_ids] = batch[14]
            feed_dict[self.pos_dayofyear_ids] = batch[15]
            feed_dict[self.pos_week_ids] = batch[16]
            feed_dict[self.pos_hour_ids] = batch[17]
            feed_dict[self.item_fism_ids] = batch[18]
            feed_dict[self.timestamp_ids] = batch[19]
            feed_dict[self.pos_timestamp_ids] = batch[20]
            feed_dict[self.inserted_seq_ids] = batch[21]
            feed_dict[self.switched_seq_ids] = batch[22]
            feed_dict[self.u_value] = batch[23]
            feed_dict[self.uneven_uniform_seq] = batch[24]
            feed_dict[self.un_year_ids] = batch[25]
            feed_dict[self.un_month_ids] = batch[26]
            feed_dict[self.un_day_ids] = batch[27]
            feed_dict[self.un_dayofweek_ids] = batch[28]
            feed_dict[self.un_dayofyear_ids] = batch[29]
            feed_dict[self.un_week_ids] = batch[30]
            feed_dict[self.un_hour_ids] = batch[31]
            feed_dict[self.uneven_item_num]=batch[32]
            feed_dict[self.item_head_tail_values]=batch[33]
            feed_dict[self.item_neighbors_sequences]=batch[34]
            feed_dict[self.item_var_values]=batch[35]
            feed_dict[self.epoch_num]=epoch_num
        else:
            feed_dict[self.test_item_ids] = batch[2]  # [N, M]
            feed_dict[self.seq_year_ids] = batch[3]
            feed_dict[self.seq_month_ids] = batch[4]
            feed_dict[self.seq_day_ids] = batch[5]
            feed_dict[self.seq_dayofweek_ids] = batch[6]
            feed_dict[self.seq_dayofyear_ids] = batch[7]
            feed_dict[self.seq_week_ids] = batch[8]
            feed_dict[self.seq_hour_ids] = batch[9]
            feed_dict[self.test_year_ids] = batch[10]
            feed_dict[self.test_month_ids] = batch[11]
            feed_dict[self.test_day_ids] = batch[12]
            feed_dict[self.test_dayofweek_ids] = batch[13]
            feed_dict[self.test_dayofyear_ids] = batch[14]
            feed_dict[self.test_week_ids] = batch[15]
            feed_dict[self.test_hour_ids] = batch[16]
            feed_dict[self.item_fism_ids] = batch[17]
            feed_dict[self.timestamp_ids] = batch[18]
            feed_dict[self.test_timestamp_ids] = batch[19]
            feed_dict[self.item_head_tail_values]=batch[20]
            feed_dict[self.item_neighbors_sequences]=batch[21]
            feed_dict[self.epoch_num]=epoch_num

        return feed_dict

    def export_embeddings(self):
        pass

    def _create_placeholders(self):
        super(DoubleBranchunirec, self)._create_placeholders()
        self.epoch_num = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name='epoch_num')
        self.seq_year_ids = tf.compat.v1.placeholder(name='seq_year_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.seq_month_ids = tf.compat.v1.placeholder(name='seq_month_ids',
                                                      dtype=tf.float32,
                                                      shape=(None, self.seqlen))
        self.seq_day_ids = tf.compat.v1.placeholder(name='seq_day_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.seq_dayofweek_ids = tf.compat.v1.placeholder(name='seq_dayofweek_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.seq_dayofyear_ids = tf.compat.v1.placeholder(name='seq_dayofyear_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.seq_week_ids = tf.compat.v1.placeholder(name='seq_week_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.seq_hour_ids = tf.compat.v1.placeholder(name='seq_hour_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_year_ids = tf.compat.v1.placeholder(name='pos_year_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_month_ids = tf.compat.v1.placeholder(name='pos_month_ids',
                                                      dtype=tf.float32,
                                                      shape=(None, self.seqlen))
        self.pos_day_ids = tf.compat.v1.placeholder(name='pos_day_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.pos_dayofweek_ids = tf.compat.v1.placeholder(name='pos_dayofweek_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.pos_dayofyear_ids = tf.compat.v1.placeholder(name='pos_dayofyear_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.pos_week_ids = tf.compat.v1.placeholder(name='pos_week_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_hour_ids = tf.compat.v1.placeholder(name='pos_hour_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.un_year_ids = tf.compat.v1.placeholder(name='un_year_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.un_month_ids = tf.compat.v1.placeholder(name='un_month_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.un_day_ids = tf.compat.v1.placeholder(name='un_day_ids',
                                                   dtype=tf.float32,
                                                   shape=(None, self.seqlen))
        self.un_dayofweek_ids = tf.compat.v1.placeholder(name='un_dayofweek_ids',
                                                         dtype=tf.float32,
                                                         shape=(None, self.seqlen))
        self.un_dayofyear_ids = tf.compat.v1.placeholder(name='un_dayofyear_ids',
                                                         dtype=tf.float32,
                                                         shape=(None, self.seqlen))
        self.un_week_ids = tf.compat.v1.placeholder(name='un_week_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.un_hour_ids = tf.compat.v1.placeholder(name='un_hour_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))

        self.test_year_ids = tf.compat.v1.placeholder(name='test_year_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])
        self.test_month_ids = tf.compat.v1.placeholder(name='test_month_ids',
                                                       dtype=tf.float32,
                                                       shape=[None])
        self.test_day_ids = tf.compat.v1.placeholder(name='test_day_ids',
                                                     dtype=tf.float32,
                                                     shape=[None])
        self.test_dayofweek_ids = tf.compat.v1.placeholder(name='test_dayofweek_ids',
                                                           dtype=tf.float32,
                                                           shape=[None])
        self.test_dayofyear_ids = tf.compat.v1.placeholder(name='test_dayofyear_ids',
                                                           dtype=tf.float32,
                                                           shape=[None])
        self.test_week_ids = tf.compat.v1.placeholder(name='test_week_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])
        self.test_hour_ids = tf.compat.v1.placeholder(name='test_hour_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])

        self.timestamp_ids = tf.compat.v1.placeholder(name='timestamp_ids',
                                                      dtype=tf.int32,
                                                      shape=(None, self.seqlen))
        self.pos_timestamp_ids = tf.compat.v1.placeholder(name='pos_timestamp_ids',
                                                          dtype=tf.int32,
                                                          shape=(None, self.seqlen))
        self.test_timestamp_ids = tf.compat.v1.placeholder(name='test_timestamp_ids', dtype=tf.int32,
                                                           shape=[None])
        # self.judge_b = tf.compat.v1.placeholder(name='judge_branch_num', dtype=tf.int32,shape=[None])

        self.item_fism_ids = tf.compat.v1.placeholder(
            name='item_fism_elem_ids', dtype=tf.int32,
            shape=[None, self.n_fism_elems])

        self.inserted_seq_ids = tf.compat.v1.placeholder(name='tihuan_seq_ids',
                                                  dtype=tf.int32,
                                                  shape=[None, self.seqlen])

        self.switched_seq_ids = tf.compat.v1.placeholder(name='jiaohuan_seq_ids',
                                                  dtype=tf.int32,
                                                  shape=[None, self.seqlen])

        self.uneven_uniform_seq = tf.compat.v1.placeholder(name='uneven_uniform_seq_ids',
                                                         dtype=tf.int32,
                                                         shape=[None, self.seqlen])
        self.uneven_item_num = tf.compat.v1.placeholder(name='uneven_item_num',
                                                           dtype=tf.float32,
                                                           shape=[None])
        self.u_value = tf.compat.v1.placeholder(dtype=tf.bool, shape=[None])
        
        self.item_head_tail_values = tf.compat.v1.placeholder(dtype=tf.bool, shape=[None, self.seqlen], name='item_head_tail_values')

        self.item_neighbors_sequences = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, self.seqlen, 3], name='item_neighbors_sequences')

        self.item_var_values = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.seqlen], name='item_var_values')
        

    def _create_variables(self, reuse=None):
        super(DoubleBranchunirec, self)._create_variables(reuse=reuse)
        with tf.compat.v1.variable_scope('embedding_tables',
                                         reuse=reuse):
            self.user_embedding_table, _ = embedding(vocab_size=self.n_users + 1,
                                                     embedding_dim=self.embedding_dim,
                                                     zero_pad=True,
                                                     use_reg=self.use_reg,
                                                     l2_reg=self.l2_emb,
                                                     scope='user_embedding_table',
                                                     reuse=reuse)
            self.position_embedding_table = embedding(vocab_size=self.seqlen,
                                                      embedding_dim=self.embedding_dim,
                                                      zero_pad=False,
                                                      use_reg=self.use_reg,
                                                      l2_reg=self.l2_emb,
                                                      scope='position_embedding_table',
                                                      reuse=reuse)

            self.sigma_noise = tf.compat.v1.Variable(0.1 * tf.ones(self.num_global_heads),
                                                     trainable=True,
                                                     name=f'sigma_noise',
                                                     dtype=tf.float32)

            self.timestamp_embedding_table = embedding(
                vocab_size=8761,  
                embedding_dim=self.embedding_dim,  
                zero_pad=False,
                use_reg=self.use_reg,
                l2_reg=self.l2_emb,
                scope='timestamp_embedding_table',
                reuse=reuse
            )

    def _create_net_inference(self, name, reuse=None):
        self.logger.debug(f'--> Create inference for {name}')
        super(DoubleBranchunirec, self)._create_net_inference(name, reuse=reuse)
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=reuse):
            self.users = tf.nn.embedding_lookup(self.user_embedding_table,
                                                self.user_ids)

            self.fism_items = tf.nn.embedding_lookup(
                self.item_embedding_table, self.item_fism_ids)

            self.user_fism_items = tf.concat([
                tf.expand_dims(self.users, axis=1), self.fism_items],
                axis=1)

            self.nonscale_input_seq = tf.nn.embedding_lookup(
                self.item_embedding_table, self.seq_ids)

            self.abs_position = self._learnable_abs_position_embedding(self.seq_ids,
                self.position_embedding_table)
            self.seq += self.abs_position
            
            self.inserted_seq = tf.nn.embedding_lookup(self.item_embedding_table,
                                                self.inserted_seq_ids)
            self.switched_seq = tf.nn.embedding_lookup(self.item_embedding_table,
                                                self.switched_seq_ids)
            self.uneven_uniform_emb = tf.nn.embedding_lookup(self.item_embedding_table,
                                                self.uneven_uniform_seq)

            if self.input_scale is True:
                self.logger.info('Scale input sequence')
                self.inserted_seq = self.inserted_seq * (self.embedding_dim ** 0.5)
                self.switched_seq = self.switched_seq * (self.embedding_dim ** 0.5)

                self.uneven_uniform_emb = self.uneven_uniform_emb * (self.embedding_dim ** 0.5)
            else:
                self.logger.info('DO NOT scale input')

            i_pos= self._learnable_abs_position_embedding(self.inserted_seq_ids,
                self.position_embedding_table)
            s_pos = self._learnable_abs_position_embedding(self.switched_seq_ids,
                                                           self.position_embedding_table)
            un_pos =  self._learnable_abs_position_embedding(self.uneven_uniform_seq,
                                                           self.position_embedding_table)
            self.inserted_seq += i_pos
            self.switched_seq += s_pos
            self.uneven_uniform_emb += un_pos

            self.ctx_seq = self._ctx_representation(reuse=reuse,
                                                    year_ids=self.seq_year_ids,
                                                    month_ids=self.seq_month_ids,
                                                    day_ids=self.seq_day_ids,
                                                    dayofweek_ids=self.seq_dayofweek_ids,
                                                    dayofyear_ids=self.seq_dayofyear_ids,
                                                    week_ids=self.seq_week_ids,
                                                    hour_ids=self.seq_hour_ids,
                                                    seqlen=self.seqlen,
                                                    use_year=self.use_year,
                                                    activation=self.ctx_activation,
                                                    name='ctx_input_seq')

            ctx_seq = tf.identity(self.ctx_seq)

            if self.input_scale is True:
                self.logger.info('Scale context sequences')
                ctx_seq = ctx_seq * (self.embedding_dim ** 0.5)

            loc_ctx_seq = ctx_seq + self.abs_position

            self.un_ctx_seq = self._ctx_representation(reuse=reuse,
                                                    year_ids=self.seq_year_ids,
                                                    month_ids=self.seq_month_ids,
                                                    day_ids=self.seq_day_ids,
                                                    dayofweek_ids=self.seq_dayofweek_ids,
                                                    dayofyear_ids=self.seq_dayofyear_ids,
                                                    week_ids=self.seq_week_ids,
                                                    hour_ids=self.seq_hour_ids,
                                                    seqlen=self.seqlen,
                                                    use_year=self.use_year,
                                                    activation=self.ctx_activation,
                                                    name='ctx_input_seq')

            un_ctx_seq = tf.identity(self.un_ctx_seq)

            if self.input_scale is True:
                self.logger.info('Scale context sequences')
                un_ctx_seq = un_ctx_seq * (self.embedding_dim ** 0.5)

            un_ctx_seq = un_ctx_seq + un_pos

            self.ctx_seq_ts = tf.nn.embedding_lookup(
                self.timestamp_embedding_table, self.timestamp_ids)

            self.ctx_seq_ts = tf.reshape(
                self.ctx_seq_ts,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])

            ctx_seq_ts = tf.identity(self.ctx_seq_ts)

            if self.input_scale is True:
                self.logger.info('Scale context sequences')
                ctx_seq_ts = ctx_seq_ts * (self.embedding_dim ** 0.5)
            loc_ctx_seq_ts = ctx_seq_ts + self.abs_position

        mask = self._get_mask()

        self.loc_seq = self._seq_representation(self.seq, loc_ctx_seq,
                                                sigma_noise=self.sigma_noise,
                                                mask=mask,
                                                reuse=reuse,
                                                causality=self.causality,
                                                name='local')

        self.loc_seq_b2 = self._seq_representation(self.seq, loc_ctx_seq_ts,
                                                   sigma_noise=self.sigma_noise,
                                                   mask=mask,
                                                   reuse=reuse,
                                                   causality=self.causality,
                                                   name='local')

        self.loc_un_seq = self._seq_representation(self.uneven_uniform_emb, un_ctx_seq,
                                                   sigma_noise=self.sigma_noise,
                                                   mask=mask,
                                                   reuse=reuse,
                                                   causality=self.causality,
                                                   name='local')

        self.loc_seq_s = tf.concat([self.inserted_seq, loc_ctx_seq], axis=-1)
        self.loc_seq_i = tf.concat([self.switched_seq, loc_ctx_seq], axis=-1)

    def _seq_representation(self, seq, ctx_seq, sigma_noise,
                            mask, reuse, causality, name=''):
        with tf.compat.v1.variable_scope(f'{name}_net_inference',
                                         reuse=reuse):
            concat_seq = tf.concat([seq, ctx_seq], axis=-1)
            out_seq = self._admix_sas_representation(
                seq=concat_seq,
                context_seq=self.ctx_seq,
                sigma_noise=sigma_noise,
                mask=mask,
                causality=causality,
                name=f'{name}_concat_seq',
                reuse=reuse)
        return out_seq

    def _seq_representation_b2(self, seq,
                            mask, reuse, causality, name=''):
        with tf.compat.v1.variable_scope(f'{name}_net_inference',
                                         reuse=reuse):
            seq = tf.compat.v1.layers.dropout(
                seq,
                rate=self.dropout_rate,
                training=tf.convert_to_tensor(self.is_training))
            seq *= mask
            out_seq = multi_head_attention_blocks(
                seq=seq,
                num_blocks=4,
                dim_head=8,
                num_heads=8,
                dropout_rate=self.dropout_rate,
                mask=mask,
                output_dim=self.embedding_dim,
                causality=causality,
                residual_type='add', 
                reuse=reuse,
                is_training=self.is_training,
                name=name
            )
            out_seq= normalize(out_seq)
        return out_seq


    def _admix_sas_representation(self, seq, context_seq,
                                  sigma_noise, mask, causality,
                                  name='', reuse=None):
        sigma_noise = tf.expand_dims(sigma_noise, axis=0)
        sigma_noise = tf.tile(sigma_noise, [self.batch_size, 1])
        seq = tf.compat.v1.layers.dropout(
            seq,
            rate=self.dropout_rate,
            training=tf.convert_to_tensor(self.is_training))
        seq *= mask
        seq = admix_multi_head_attention_blocks(seq=seq,
                                                context_seq=context_seq,
                                                num_blocks=self.num_blocks,
                                                num_heads=self.num_heads,
                                                dim_head=self.dim_head,
                                                sigma_noise=sigma_noise,
                                                dropout_rate=self.dropout_rate,
                                                mask=mask,
                                                output_dim=self.local_output_dim,
                                                causality=causality,
                                                residual_type=self.residual_type,
                                                is_training=self.is_training,
                                                reuse=reuse,
                                                name=f'{name}_mha_blocks')
        seq = normalize(seq)
        return seq

    def _ctx_representation(self, year_ids, month_ids,
                            day_ids, dayofweek_ids,
                            dayofyear_ids, week_ids, hour_ids,
                            seqlen, reuse, use_year=True,
                            activation=None,
                            name='shared_context_representation'):

        seq_years = self.basis_time_encode(inputs=year_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='year',
                                           reuse=reuse)
        seq_months = self.basis_time_encode(inputs=month_ids,
                                            time_dim=self.tempo_embedding_dim,
                                            expand_dim=self.expand_dim,
                                            scope='month',
                                            reuse=reuse)
        seq_days = self.basis_time_encode(inputs=day_ids,
                                          time_dim=self.tempo_embedding_dim,
                                          expand_dim=self.expand_dim,
                                          scope='day',
                                          reuse=reuse)
        seq_dayofweeks = self.basis_time_encode(inputs=dayofweek_ids,
                                                time_dim=self.tempo_embedding_dim,
                                                expand_dim=self.expand_dim,
                                                scope='dayofweek',
                                                reuse=reuse)
        seq_dayofyears = self.basis_time_encode(inputs=dayofyear_ids,
                                                time_dim=self.tempo_embedding_dim,
                                                expand_dim=self.expand_dim,
                                                scope='dayofyear',
                                                reuse=reuse)
        seq_weeks = self.basis_time_encode(inputs=week_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='week',
                                           reuse=reuse)
        seq_hours = self.basis_time_encode(inputs=hour_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='hour',
                                           reuse=reuse)
        if use_year is True:
            ctx_seq_concat = tf.concat([seq_years, seq_months,
                                        seq_days, seq_dayofweeks,
                                        seq_dayofyears, seq_weeks, seq_hours],
                                       axis=-1)
            ctx_seq_concat = tf.reshape(
                ctx_seq_concat,
                shape=[tf.shape(self.seq_ids)[0] * seqlen,
                       self.num_contexts * self.tempo_embedding_dim])
        else:
            ctx_seq_concat = tf.concat([seq_months,
                                        seq_days, seq_dayofweeks,
                                        seq_dayofyears, seq_weeks, seq_hours], axis=-1)
            ctx_seq_concat = tf.reshape(
                ctx_seq_concat,
                shape=[tf.shape(self.seq_ids)[0] * seqlen,
                       (self.num_contexts - 1) * self.tempo_embedding_dim])

        ctx_seq = tf.compat.v1.layers.dense(
            inputs=ctx_seq_concat,
            units=self.embedding_dim,
            activation=activation,
            reuse=reuse,
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.01), name=f'{name}_dense_output')
        ctx_seq = tf.compat.v1.layers.dropout(
            ctx_seq,
            rate=self.dropout_rate,
            training=tf.convert_to_tensor(self.is_training))
        ctx_seq = tf.reshape(
            ctx_seq,
            shape=[tf.shape(self.seq_ids)[0], seqlen, self.embedding_dim],
            name=f'{name}_context_embComp')
        return ctx_seq

    def _create_test_inference(self, name, reuse=None):

        test_item_emb = tf.nn.embedding_lookup(self.item_embedding_table,
                                               self.test_item_ids)
        test_ctx_seq = self._test_context_seq(reuse=tf.compat.v1.AUTO_REUSE)
        fused_test_item_emb = tf.concat([test_item_emb, test_ctx_seq],
                                        axis=-1)

        self.loc_test_logits = tf.matmul(self.loc_seq,
                                         tf.transpose(fused_test_item_emb, perm=[0, 2, 1]))

        test_ctx_seq_ts = tf.nn.embedding_lookup(self.timestamp_embedding_table, self.test_timestamp_ids)
        test_ctx_seq_ts = tf.tile(tf.expand_dims(test_ctx_seq_ts, 1), [1, self.num_test_negatives + 1, 1])
        fused_test_item_emb_b2=tf.concat([test_item_emb, test_ctx_seq_ts],
                                        axis=-1)
        self.loc_b2_test_logits = tf.matmul(self.loc_seq_b2,
                                         tf.transpose(fused_test_item_emb_b2, perm=[0, 2, 1]))

        if self.lambda_glob > 0:
            att_seq = self._fism_attentive_vectors(self.user_fism_items,
                                                   self.nonscale_input_seq)
            glob_seq_vecs = self.nonscale_input_seq * (1.0 - self.lambda_trans_seq) + \
                            (self.nonscale_input_seq * att_seq) * self.lambda_trans_seq
            glob_seq_vecs = tf.reduce_sum(glob_seq_vecs[:, 1:, :], axis=1,
                                          keepdims=True)
            glob_test_atts = self._fism_attentive_vectors(self.user_fism_items,
                                                          test_item_emb)
            glob_test_logits = test_item_emb * (1.0 - self.lambda_trans_seq) + \
                               (test_item_emb * glob_test_atts) * self.lambda_trans_seq
            glob_test_logits = (glob_test_logits + glob_seq_vecs) / self.seqlen
            glob_test_logits = tf.reduce_sum(glob_test_logits * test_item_emb,
                                             axis=-1)
            loc_test_logits = self.loc_test_logits[:, -1, :]
            loc_test_logits_b2 = self.loc_b2_test_logits[:, -1, :]

            self.test_logits = loc_test_logits + self.lambda_glob * glob_test_logits + loc_test_logits_b2
        else:
            self.test_logits = self.loc_test_logits + self.loc_b2_test_logits

            self.test_logits = self.test_logits[:, -1, :]

    def _test_context_seq(self, reuse=None):
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=reuse):
            test_year_ids = tf.tile(tf.expand_dims(self.test_year_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_month_ids = tf.tile(tf.expand_dims(self.test_month_ids, axis=-1),
                                     [1, self.num_test_negatives + 1])
            test_day_ids = tf.tile(tf.expand_dims(self.test_day_ids, axis=-1),
                                   [1, self.num_test_negatives + 1])
            test_dayofweek_ids = tf.tile(tf.expand_dims(self.test_dayofweek_ids, axis=-1),
                                         [1, self.num_test_negatives + 1])
            test_dayofyear_ids = tf.tile(tf.expand_dims(self.test_dayofyear_ids, axis=-1),
                                         [1, self.num_test_negatives + 1])
            test_week_ids = tf.tile(tf.expand_dims(self.test_week_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_hour_ids = tf.tile(tf.expand_dims(self.test_hour_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_ctx_seq = self._ctx_representation(
                year_ids=test_year_ids, month_ids=test_month_ids,
                day_ids=test_day_ids, dayofweek_ids=test_dayofweek_ids,
                dayofyear_ids=test_dayofyear_ids, week_ids=test_week_ids,
                hour_ids=test_hour_ids,
                seqlen=self.num_test_negatives + 1,
                reuse=tf.compat.v1.AUTO_REUSE,
                use_year=self.use_year,
                activation=self.ctx_activation,
                name='ctx_input_seq')

        return test_ctx_seq

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        loc_loss = self._loss_on_seq(self.loc_seq)
        loc_loss_b2=self._loss_on_seq_b2(self.loc_seq_b2)
        if self.lambda_glob > 0:  
            pos_seq = tf.reshape(
                self.pos_emb,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            neg_seq = tf.reshape(
                self.neg_emb,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            if self.lambda_trans_seq > 0:
                pos_att_vecs = self._adaptive_attentive_seq(
                    self.pos_emb,
                    user_fism_items=self.user_fism_items,
                    need_reshaped=True,
                    name='adaptive_pos_sequence')
            else:
                pos_att_vecs = tf.reshape(
                    self.pos_emb,
                    shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            pos_logits = tf.reshape(
                tf.reduce_sum(pos_att_vecs * pos_seq, axis=-1),
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen])

            if self.lambda_trans_seq > 0:
                neg_att_vecs = self._adaptive_attentive_seq(
                    self.neg_emb,
                    user_fism_items=self.user_fism_items,
                    need_reshaped=True,
                    name='adaptive_neg_sequence')
            else:
                neg_att_vecs = tf.reshape(
                    self.neg_emb,
                    shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            neg_logits = tf.reshape(
                tf.reduce_sum(neg_att_vecs * neg_seq, axis=-1),
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen])
            # Regularization
            l2_norm = tf.add_n([
                self.lambda_user * tf.reduce_sum(tf.multiply(
                    self.user_embedding_table, self.user_embedding_table)),
                self.lambda_item * tf.reduce_sum(tf.multiply(
                    self.item_embedding_table, self.item_embedding_table))
            ])
            glob_loss = tf.reduce_sum(
                - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) * self.istarget -
                tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24) * self.istarget
            ) / tf.reduce_sum(self.istarget) + l2_norm
            self.loss = loc_loss + self.lambda_glob * glob_loss + loc_loss_b2
        else:
            self.loss = loc_loss + loc_loss_b2

        #userbranch
        self.user_branch_loss= self.get_userbranch_loss()
        self.loss+=self.user_branch_loss

        # itembranch
        self.item_branch_loss = self.item_branch_loss()
        self.loss+=self.item_branch_loss

        self.item_tail_loss = self.tail_item_loss()
        self.loss += self.item_tail_loss

        def conditional_update():
            user_branch_loss = self.get_userbranch_loss()
            return self.loss + user_branch_loss
        
        def no_update():
            return self.loss
        
        self.loss = tf.cond(tf.greater(self.epoch_num, 99), conditional_update, no_update)

        self.reg_loss = sum(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) 
        self.loss += self.reg_loss

    def _loss_on_seq(self, seq): 
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=tf.compat.v1.AUTO_REUSE):

            seq_emb = tf.reshape(
                seq,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.local_output_dim])

            pos_ctx_seq = self._ctx_representation(
                year_ids=self.pos_year_ids, month_ids=self.pos_month_ids,
                day_ids=self.pos_day_ids, dayofweek_ids=self.pos_dayofweek_ids,
                dayofyear_ids=self.pos_dayofyear_ids, week_ids=self.pos_week_ids,
                hour_ids=self.pos_hour_ids,
                seqlen=self.seqlen,
                reuse=tf.compat.v1.AUTO_REUSE,
                use_year=self.use_year,
                activation=self.ctx_activation,
                name='ctx_input_seq')

            ctx_emb = tf.reshape(
                pos_ctx_seq,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.embedding_dim])


        pos_emb = tf.concat([self.pos_emb, ctx_emb], axis=-1)
        neg_emb = tf.concat([self.neg_emb, ctx_emb], axis=-1)

        # prediction layer
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(pos_logits) + 1e-8) * self.istarget -
            tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-8) * self.istarget
        ) / tf.reduce_sum(self.istarget)

        return loss

    def _loss_on_seq_b2(self, seq):  
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=tf.compat.v1.AUTO_REUSE):

            seq_emb = tf.reshape(
                seq,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.local_output_dim])

            pos_ctx_seq_ts = tf.nn.embedding_lookup(self.timestamp_embedding_table, self.pos_timestamp_ids)
            ctx_emb_ts = tf.reshape(
                pos_ctx_seq_ts,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.embedding_dim])
            ctx_emb = ctx_emb_ts

        pos_emb = tf.concat([self.pos_emb, ctx_emb], axis=-1)
        neg_emb = tf.concat([self.neg_emb, ctx_emb], axis=-1)

        # prediction layer
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(pos_logits) + 1e-8) * self.istarget -
            tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-8) * self.istarget
        ) / tf.reduce_sum(self.istarget)

        return loss

    def basis_time_encode(self, inputs, time_dim, expand_dim,
                          scope='basis_time_kernel', reuse=None,
                          return_weight=False):
        """Mercer's time encoding

        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          time_dim: integer, number of dimention for time embedding
          expand_dim: degree of frequency expansion
          scope: string, scope for tensorflow variables
          reuse: bool, if true the layer could be reused
          return_weight: bool, if true return both embeddings and frequency

        Returns:
          A 3d float tensor which embeds the input or
          A tuple with one 3d float tensor (embeddings) and 2d float tensor (frequency)
        """

        # inputs: [N, max_len]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            expand_input = tf.tile(tf.expand_dims(inputs, 2),
                                   [1, 1, time_dim])  # [N, max_len, time_dim]
            init_period_base = np.linspace(0, self.tempo_linspace, time_dim)
            init_period_base = init_period_base.astype(np.float32)
            period_var = tf.compat.v1.get_variable('time_cos_freq',
                                                   dtype=tf.float32,
                                                   initializer=tf.constant(init_period_base))
            period_var = 10.0 ** period_var
            period_var = tf.tile(tf.expand_dims(period_var, 1),
                                 [1, expand_dim])  # [time_dim] -> [time_dim, 1] -> [time_dim, expand_dim]
            expand_coef = tf.cast(tf.reshape(tf.range(expand_dim) + 1, [1, -1]), tf.float32)

            freq_var = 1 / period_var
            freq_var = freq_var * expand_coef

            basis_expan_var = tf.compat.v1.get_variable(
                'basis_expan_var',
                shape=[time_dim, 2 * expand_dim],
                initializer=tf.compat.v1.glorot_uniform_initializer())

            basis_expan_var_bias = tf.compat.v1.get_variable(
                'basis_expan_var_bias',
                shape=[time_dim],
                initializer=tf.zeros_initializer)  # initializer=tf.glorot_uniform_initializer())

            sin_enc = tf.sin(tf.multiply(tf.expand_dims(expand_input, -1),
                                         tf.expand_dims(tf.expand_dims(freq_var, 0), 0)))
            cos_enc = tf.cos(tf.multiply(tf.expand_dims(expand_input, -1),
                                         tf.expand_dims(tf.expand_dims(freq_var, 0), 0)))
            time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1),
                                   tf.expand_dims(tf.expand_dims(basis_expan_var, 0), 0))
            time_enc = tf.add(tf.reduce_sum(time_enc, -1),
                              tf.expand_dims(tf.expand_dims(basis_expan_var_bias, 0), 0))

        if return_weight:
            return time_enc, freq_var
        return time_enc

    def _adaptive_attentive_seq(self, seq, user_fism_items,
                                name='', need_reshaped=True):
        if need_reshaped is True:
            seq = tf.reshape(
                seq,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
        att_seq = self._fism_attentive_vectors(user_fism_items, seq,
                                               name=name)
        if self.lambda_trans_seq < 1:
            att_fism_seq = seq * (1.0 - self.lambda_trans_seq) + \
                           (seq * att_seq) * self.lambda_trans_seq
        else:
            att_fism_seq = seq * att_seq
        return att_fism_seq

    def _fism_attentive_vectors(self, fism_items, seq, name=''): 
        with tf.name_scope(name):
            w_ij = tf.matmul(seq,
                             tf.transpose(fism_items, perm=[0, 2, 1]))  
            exp_wij = tf.exp(w_ij) 
            exp_sum = tf.reduce_sum(exp_wij, axis=-1, keepdims=True)  
            if self.beta != 1.0:
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            att = exp_wij / exp_sum 
            att_vecs = tf.matmul(att, fism_items) 
        return att_vecs

    def calculate_attention_scores(self, embedding1, embedding2):
        combined_embeddings = tf.concat([embedding1, embedding2], axis=-1)  
        attention_network = tf.keras.layers.Dense(1,
                                                  activation='sigmoid') 
        attention_scores = attention_network(combined_embeddings)  
        return attention_scores

    def choose_embeddings(self, embedding1, embedding2, attention_scores):
    
        chosen_embedding = attention_scores * embedding2 + (1 - attention_scores) * embedding1
        return chosen_embedding

    def get_userbranch_loss(self):
      
        u_value = tf.tile(tf.expand_dims(self.u_value, -1), [1, self.seqlen])
        expanded_u_value = tf.expand_dims(u_value, -1)
        u_value = tf.tile(expanded_u_value, [1, 1, self.local_output_dim])
        u_value = tf.cast(u_value, tf.float32)
        w_u = tf.cast((np.pi / 2), tf.float32) * tf.cast(((self.epoch_num - 100) / 120), tf.float32) + tf.cast(
            (np.pi / 2), tf.float32) * tf.cast(((self.uneven_item_num - 3) / 50), tf.float32)
        w_u = tf.math.abs(tf.math.sin(w_u))
       
        w_u = tf.reshape(w_u, [self.batch_size, 1, 1])

        W = add_weight(self.local_output_dim, name='linear_weight')
        uneven_emb = tf.reshape(self.loc_un_seq, [-1, self.local_output_dim])
        linear_output = tf.matmul(uneven_emb, W)
        linear_output_reshaped = tf.reshape(linear_output, [self.batch_size, self.seqlen, self.local_output_dim])

        user_branch_loss =  w_u * ((linear_output_reshaped - self.loc_seq) ** 2)
        user_branch_loss = u_value * user_branch_loss
        loss=tf.reduce_mean(user_branch_loss)*0.2
        return loss

    def item_neighbors_combine(self, seq):
       
        item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, seq)

      
        flat_neighbor_ids = tf.reshape(self.item_neighbors_sequences, [-1])

        flat_neighbor_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, flat_neighbor_ids)

        neighbor_embeddings = tf.reshape(flat_neighbor_embeddings, [-1, tf.shape(seq)[1], 3, self.embedding_dim])

        expanded_item_embeddings = tf.expand_dims(item_embeddings, axis=2)  

        dot_product = tf.reduce_sum(expanded_item_embeddings * neighbor_embeddings, axis=-1)  
        exp_dot_product = tf.exp(dot_product)
        weights = exp_dot_product / tf.reduce_sum(exp_dot_product, axis=-1, keepdims=True)  

        weighted_neighbor_embeddings = tf.reduce_sum(neighbor_embeddings * tf.expand_dims(weights, axis=-1), axis=2)  

        all_embeddings = tf.concat([item_embeddings, weighted_neighbor_embeddings], axis=-1) 

        return all_embeddings



    def item_branch_loss(self):
        item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.seq_ids)

        flat_neighbor_ids = tf.reshape(self.item_neighbors_sequences, [-1])

        flat_neighbor_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, flat_neighbor_ids)

        neighbor_embeddings = tf.reshape(flat_neighbor_embeddings, [-1, tf.shape(self.seq_ids)[1], 3, self.embedding_dim])

        expanded_item_embeddings = tf.expand_dims(item_embeddings, axis=2) 

        dot_product = tf.reduce_sum(expanded_item_embeddings * neighbor_embeddings, axis=-1) 
        exp_dot_product = tf.exp(dot_product)
        weights = exp_dot_product / tf.reduce_sum(exp_dot_product, axis=-1, keepdims=True) 

        weighted_neighbor_embeddings = tf.reduce_sum(neighbor_embeddings * tf.expand_dims(weights, axis=-1), axis=2) 

        all_embeddings = tf.concat([item_embeddings, weighted_neighbor_embeddings], axis=2) 

        reshaped_embeddings = tf.reshape(all_embeddings, [-1, 2 * self.embedding_dim])

        head_item_mask = tf.cast(self.item_head_tail_values, tf.float32)
        head_item_mask = tf.reshape(head_item_mask, [-1, 1]) 
        masked_combined_embeddings = reshaped_embeddings * head_item_mask 

        final_embeddings = linear_layer(masked_combined_embeddings, self.embedding_dim, trainable=True, name="linear_layer")
      
        final_embeddings = tf.reshape(final_embeddings, [self.batch_size, self.seqlen, self.embedding_dim])

        head_item_seq = tf.cast(self.item_head_tail_values, tf.float32)
        head_item_mask = tf.expand_dims(head_item_seq, -1)
        head_item_embeddings = item_embeddings * head_item_mask

        coefficients = (tf.constant(np.pi / 2) * (tf.cast(self.epoch_num, tf.float32) - 100.0) / 120.0) + \
               (tf.constant(np.pi / 2) * (100.0 - self.item_var_values) / 100.0) #[batch_size, self.seqlen]

        sin_coefficients = tf.sin(coefficients)
        w_i = tf.expand_dims(sin_coefficients, -1)

        item_branch_loss =  w_i * ((final_embeddings - head_item_embeddings) ** 2)
        loss = tf.reduce_mean(item_branch_loss)*0.3

        return loss

    def tail_item_loss(self):
        self.neighbors_embedding = self.item_neighbors_combine(self.seq_ids)
        item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.seq_ids)
        final_embeddings = linear_layer(self.neighbors_embedding, self.embedding_dim, trainable=False, name="linear_layer")
        final_embeddings = tf.reshape(final_embeddings, [self.batch_size, self.seqlen, self.embedding_dim])
        updated_seq = tf.where(tf.expand_dims(self.item_head_tail_values, axis=-1), item_embeddings, final_embeddings)
        item_tail_loss = (updated_seq - item_embeddings) ** 2
        loss = tf.reduce_mean(item_tail_loss) * 0.1 
        sin_w= tf.sin(tf.constant(np.pi/2)*(tf.cast(self.epoch_num,tf.float32)/220))
        loss = loss * sin_w

        return loss

