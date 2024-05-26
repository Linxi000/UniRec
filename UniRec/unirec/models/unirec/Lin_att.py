import tensorflow as tf
from unirec.models.core.net import feedforward, normalize

def multihead_attention(queries,
                        keys,
                        num_heads=8,
                        dim_head=16,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        residual_type='add',
                        reuse=None,
                        with_att=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Dense layers for Q, K, V
        Q = tf.compat.v1.layers.dense(queries, num_heads * dim_head, activation=None)
        K = tf.compat.v1.layers.dense(keys, num_heads * dim_head, activation=None)
        V = tf.compat.v1.layers.dense(keys, num_heads * dim_head, activation=None)

        # Apply ELU activation
        elu = tf.nn.elu
        Q = elu(Q)
        K = elu(K)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Normalization
        Q_norm = tf.nn.l2_normalize(Q_, axis=-1)
        K_norm = tf.nn.l2_normalize(K_, axis=-1)

        # Elu Norm Attention Score Calculation
        att_scores = tf.matmul(Q_norm, tf.transpose(K_norm, [0, 2, 1])) / tf.sqrt(float(dim_head))

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(att_scores) * (-2 ** 32 + 1)
        att_scores = tf.where(tf.equal(key_masks, 0), paddings, att_scores)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(att_scores[0, :, :])  # (T_q, T_k)
            tril = tf.compat.v1.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(att_scores)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            att_scores = tf.where(tf.equal(masks, 0), paddings, att_scores)  # (h*N, T_q, T_k)

        # Activation
        att_scores = tf.nn.softmax(att_scores)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        att_scores *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        att_scores = tf.compat.v1.layers.dropout(att_scores, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(att_scores, V_)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)



        # Residual connection
        if residual_type == 'add':
            outputs += queries
        elif residual_type == 'mult':
            outputs *= queries
        else:
            raise ValueError(f'Unsupported residual type {residual_type}')

    if with_att:
        return outputs, att_scores
    else:
        return outputs


def multi_head_attention_blocks(seq, num_blocks, dim_head, num_heads, dropout_rate, mask, output_dim=-1, causality=True,
                                residual_type='add', reuse=None, is_training=False, name=''):
    embedding_dim = num_heads * dim_head #64
    for i in range(num_blocks):
        with tf.compat.v1.variable_scope(f'{name}_num_blocks_{i}'):
            # Self-attention
            seq = multihead_attention(
                queries=normalize(seq),
                keys=seq,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout_rate=dropout_rate,
                is_training=is_training,
                causality=causality,
                residual_type=residual_type,
                reuse=reuse,
                scope=f'{name}_self_attention{i}')
            if i == num_blocks - 1 and output_dim > 0:
                num_units = [embedding_dim, output_dim]
            else:
                num_units = [embedding_dim, embedding_dim]
            # Feed forward net
            seq = feedforward(normalize(seq), num_units=num_units, dropout_rate=dropout_rate, is_training=is_training)
            seq *= mask
    return seq
