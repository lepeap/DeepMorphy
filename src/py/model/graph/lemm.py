
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from graph.base import GraphPartBase


class Lemm(GraphPartBase):

    def __init__(self, for_usage, global_settings, current_settings, optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, 'lemm', ["Loss", "AccuracyByChar", "Accuracy"])
        self.chars_count = self.chars_count + 1
        self.start_char_index = global_settings['start_token']
        self.end_char_index = global_settings['end_token']
        self.results = []
        self.ys = []
        self.y_seq_lens = []
        self.cls = []
        self.keep_drops = []
        self.decoder_keep_drops = []
        self.sampling_probability = None
        self.sampling_probability_value = current_settings['sampling_probability']

    def __build_graph_for_device__(self, x, seq_len, batch_size, cls=None):
        self.xs.append(x)
        self.seq_lens.append(seq_len)

        if cls is None:
            cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')

        if batch_size is None:
            batch_size = self.settings['batch_size']

        y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Y')
        y_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='YSeqLen')
        if self.sampling_probability is None:
            self.sampling_probability = tf.placeholder(tf.float32, [], name='SamplingProbability')

        self.ys.append(y)
        self.y_seq_lens.append(y_seq_len)
        start_tokens = tf.fill([batch_size], self.start_char_index)
        y_seq_len += 1

        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            encoder_char_embeddings = tf.get_variable("CharEmbeddings", [self.chars_count, self.settings['char_vector_size']])
            encoder_cls_init = tf.random_normal((self.main_classes_count, self.settings['encoder']['rnn_state_size']))
            encoder_cls_emb = tf.get_variable("ClsEmbeddings", initializer=encoder_cls_init)
            encoder_init_state = tf.nn.embedding_lookup(encoder_cls_emb, cls)
            encoder_input = tf.nn.embedding_lookup(encoder_char_embeddings, x)
#
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            decoder_char_embeddings = tf.get_variable("CharEmbeddings", [self.chars_count, self.settings['char_vector_size']])
            decoder_cls_init = tf.random_normal((self.main_classes_count, self.settings['decoder']['rnn_state_size']))
            decoder_cls_emb = tf.get_variable("ClsEmbeddings", initializer=decoder_cls_init)
            decoder_init_state = tf.nn.embedding_lookup(decoder_cls_emb, cls)
            decoder_output = tf.nn.embedding_lookup(decoder_char_embeddings, x)

        if self.for_usage:
            keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
            decoder_keep_drop = tf.constant(1, dtype=tf.float32, name='DecoderKeepDrop')
        else:
            keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')
            decoder_keep_drop = tf.placeholder(dtype=tf.float32, name='DecoderKeepDrop')

        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
            _, encoder_output = tfu.build_rnn(
                encoder_input,
                keep_drop,
                seq_len,
                self.settings['encoder'],
                encoder_init_state,
                encoder_init_state,
                top_concat=False,
                for_usage=self.for_usage,
                with_seq=True
            )


        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:

            if self.for_usage:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_char_embeddings,
                                                                 start_tokens=start_tokens,
                                                                 end_token=self.end_char_index)
            else:
                start_tokens_emd = tf.nn.embedding_lookup(decoder_char_embeddings, start_tokens)
                start_tokens_emd = tf.reshape(start_tokens_emd, (batch_size, -1, self.settings['char_vector_size']))
                encoder_output = tf.concat(values=[start_tokens_emd, encoder_output], axis=1)

                end_tokens = tf.fill([batch_size], self.end_char_index)
                end_tokens_emd = tf.nn.embedding_lookup(decoder_char_embeddings, end_tokens)
                end_tokens_emd = tf.reshape(end_tokens_emd, (batch_size, -1, self.settings['char_vector_size']))
                decoder_output = tf.concat([decoder_output, end_tokens_emd], axis=1)
                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_output,
                                                                          y_seq_len,
                                                                          decoder_char_embeddings,
                                                                          self.sampling_probability)

                end_tokens = tf.reshape(end_tokens, (batch_size, 1))
                y = tf.concat([y, end_tokens], axis=1)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.settings['decoder']['rnn_state_size'],
                memory=encoder_output,
                memory_sequence_length=seq_len
            )

            cell = tfu.rnn_cell(self.settings['decoder'],
                                self.for_usage,
                                decoder_keep_drop)

            cell = tf.contrib.seq2seq.AttentionWrapper(
               cell,
               attention_mechanism,
               attention_layer_size=self.settings['decoder']['rnn_state_size']/2
            )

            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                cell,
                self.chars_count,
            )

            init_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size) \
                .clone(cell_state=decoder_init_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell,
                helper=helper,
                initial_state=init_state
            )

            max_len = self.settings['max_length'] if self.for_usage else tf.reduce_max(y_seq_len)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False if self.for_usage else True,
                output_time_major=False,
                maximum_iterations=max_len
            )
            decoder_ids = outputs[0].sample_id

            if not self.for_usage:
                decoder_ids = outputs[0].sample_id
                self.prints.append(decoder_ids)
                decoder_logits = outputs[0].rnn_output
                self.prints.append(decoder_logits)

                masks = tf.sequence_mask(
                    lengths=y_seq_len,
                    dtype=tf.float32,
                    maxlen=tf.reduce_max(y_seq_len)
                )
                seq_mask_int = tf.cast(masks, tf.int32)
                seq_mask_flat = tf.cast(tf.reshape(masks, (-1,)), tf.int32)

                # seq loss
                loss = tf.contrib.seq2seq.sequence_loss(
                    decoder_logits,
                    y,
                    masks,
                    name="SeqLoss"
                )
                vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope_name)
                grads = self.optimiser.compute_gradients(loss, var_list=vars)
                self.__create_mean_metric__(0, loss)

                # char accuracy
                nonzero_indices = tf.where(tf.not_equal(seq_mask_flat, 0))
                labels_flat = tf.reshape(y, (-1,))
                labels_flat = tf.gather(labels_flat, nonzero_indices)
                labels_flat = tf.reshape(labels_flat, (-1,))
                predictions_flat = tf.reshape(decoder_ids, (-1,))
                predictions_flat = tf.gather(predictions_flat, nonzero_indices)
                predictions_flat = tf.reshape(predictions_flat, (-1,))
                # remove -1 items where no sampling took place
                sample_indexes = tf.where(tf.not_equal(predictions_flat, -1))
                labels_flat = tf.gather(labels_flat, sample_indexes)
                predictions_flat = tf.gather(predictions_flat, sample_indexes)

                self.__create_accuracy_metric__(1, labels_flat, predictions_flat)

                # seq accuracy
                labels = y * seq_mask_int
                predictions = decoder_ids * seq_mask_int
                labels_flat = tf.reshape(labels, (-1,))
                predictions_flat = tf.reshape(predictions, (-1,))
                sample_zeros = tf.cast(tf.not_equal(predictions_flat, -1), tf.int32)
                predictions = predictions_flat * sample_zeros
                labels = labels_flat * sample_zeros
                predictions = tf.reshape(predictions, (batch_size, max_len))
                labels = tf.reshape(labels, (batch_size, max_len))
                delta = labels - predictions
                labels = tf.reduce_sum(delta * delta, 1)
                predictions = tf.zeros(batch_size)
                self.__create_accuracy_metric__(2, labels, predictions)

                self.dev_grads.append(grads)
                self.losses.append(loss)
                self.keep_drops.append(keep_drop)
                self.decoder_keep_drops.append(decoder_keep_drop)

            self.cls.append(cls)
            self.results.append(decoder_ids)

    #def __before_finish__(self):
    #    self.decay_step += 1
    #    if self.decay_step == 3:
    #        return True
    #    else:
    #        tqdm.write("Sampling probobality dacayed")
    #        self.sampling_probability_value = self.sampling_probability_value * 2
    #        self.__init_learn_params__()
    #        return False


    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        feed_dict[self.cls[dev_num]] = batch['x_cls']
        feed_dict[self.ys[dev_num]] = batch['y']
        feed_dict[self.y_seq_lens[dev_num]] = batch['y_seq_len']
        feed_dict[self.keep_drops[dev_num]] = self.settings['keep_drop']
        feed_dict[self.decoder_keep_drops[dev_num]] = self.settings['decoder']['keep_drop']
        feed_dict[self.sampling_probability] = self.sampling_probability_value

    def __load_dataset__(self, operation_name):
        return list(
            tfu.load_lemma_dataset(
                self.dataset_path,
                self.devices_count,
                operation_name,
                self.settings['batch_size']
            )
        )


