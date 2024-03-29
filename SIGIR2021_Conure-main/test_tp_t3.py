import tensorflow as tf
import data_loader_neg as data_loader
import generator_prune_regbig as generator_recsys
import data_analysis
import metric_evaluation
import math
import numpy as np
import argparse
import config
import sys
import os

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t


def random_negs(l,r,no,s):
    # set_s=set(s)
    negs = []
    for i in range(no):
        t = np.random.randint(l, r)
        # while (t in set_s):
        while (t== s):
            t = np.random.randint(l, r)
        negs.append(t)
    return negs


def main(path, case, extention, prune_percentage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/original_desen_finetune_like_nouserID.csv ',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=1,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=500,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--rho', type=float, default=0.3,
                        help='static sampling in LambdaFM paper')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')
    parser.add_argument('--max_position', type=int, default=1000,
                         help='maximum number of for positional embedding, it has to be larger than the sequence lens')
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--case', type=int, default=case)
    parser.add_argument('--extention', type=int, default=extention)
    parser.add_argument('--prune_percentage', type=float, default=prune_percentage)
    args = parser.parse_args()
   
    # Resetting graph
    tf.reset_default_graph()

    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath, 'dir_name_index': args.datapath_index, 'lambdafm_rho': args.rho})

    items = dl.item_dict
    items_len = len(items)
    targets = dl.target_dict
    targets_len=len(targets)
    targets_len_nozero = targets_len - 1
    bigemb = dl.embed_len
    top_k=args.top_k
    all_samples = dl.example

    #Doing Data analysis
    data_analysis.Data_Analysis({'data': all_samples,'datapath': args.datapath, 'task': 'T3', 'path': args.path})

    # Randomly shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    model_para = {
        'item_size': len(items),
        'bigemb':bigemb,
        'dilated_channels': 256,
        'target_item_size': targets_len,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':256,
        'iterations':3,
        'has_positionalembedding': args.has_positionalembedding,
        'max_position': args.max_position,
        'is_negsample':True, #False denotes using full softmax
        'taskID':config.taskID_3rd#the second task indexing from 10001
    }

    sess = tf.Session()
    taskID=model_para['taskID']
    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph()
    for index in range(taskID - config.taskID_1st):
        t_name = config.taskID_1st + index
        if index == 0:
            softmax_w = tf.get_variable("softmax_w_{}".format(t_name), [model_para['item_size'], model_para['dilated_channels']], tf.float32, tf.random_normal_initializer(0.0, 0.01))
            softmax_b = tf.get_variable("softmax_b_{}".format(t_name), [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))
        else:
            softmax_size=config.task_conf['task_itemsize'][index]
            #softmax_size = 181  ----- dataset 1 ----- -> dataset = 15 lines (1,5 batches) - no test/train set
            softmax_w = tf.get_variable("softmax_w_{}".format(t_name), [softmax_size, model_para['dilated_channels']], tf.float32, tf.random_normal_initializer(0.0, 0.01))

    init = tf.global_variables_initializer()
    trainable_vars = tf.trainable_variables()
    allable_vars=tf.all_variables()
    softmax_name_curtask="softmax_w_{}".format(taskID)
    variables_to_restore = [v for v in trainable_vars if v.name.find(softmax_name_curtask) == -1]

    weight=[v for v in trainable_vars if v.name.find("weight") != -1]
    bias=[v for v in allable_vars if v.name.find("bias") != -1]
    mask_var_all=[v for v in allable_vars if v.name.find("mask_val") != -1]#all mask
    mask_var = [v for v in mask_var_all if v.name.find(str(taskID) + "_mask_val") == -1]#previous

    ln_var_all = [v for v in trainable_vars if v.name.find("layer_norm") != -1]
    softmax_var=[v for v in trainable_vars if v.name.find(softmax_name_curtask) != -1]#current softtmax
    variables_to_restore.extend(bias)
    variables_to_restore.extend(mask_var)

    sess.run(init)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, os.path.join(path, 'T2', 'finetune') + "/model_nextitnet_transfer.ckpt")
    
    source_item_embedding = itemrec.dilate_input
    source_item_embedding = tf.reduce_mean(source_item_embedding[:, -1:, :], 1)  # use the last token
    embedding_size = tf.shape(source_item_embedding)[-1]

    with tf.variable_scope("target-item"):
        allitem_embeddings_target = itemrec.allitem_embeddings_out  # only difference
        is_training = tf.placeholder(tf.bool, shape=())

        # training
        itemseq_input_target_pos = tf.placeholder('int32', [None, None], name='itemseq_input_pos')
        itemseq_input_target_neg = tf.placeholder('int32', [None, None], name='itemseq_input_neg')
        target_item_embedding_pos = tf.nn.embedding_lookup(allitem_embeddings_target, itemseq_input_target_pos, name="target_item_embedding_pos")
        target_item_embedding_neg = tf.nn.embedding_lookup(allitem_embeddings_target, itemseq_input_target_neg, name="target_item_embedding_neg")

        pos_score = source_item_embedding * tf.reshape(target_item_embedding_pos, [-1, embedding_size])
        neg_score = source_item_embedding * tf.reshape(target_item_embedding_neg, [-1, embedding_size])
        pos_logits = tf.reduce_sum(pos_score, -1)
        neg_logits = tf.reduce_sum(neg_score, -1)

        # testing
        logits_2D = tf.matmul(source_item_embedding, tf.transpose(allitem_embeddings_target))
        top_k_test = tf.nn.top_k(logits_2D, k=args.top_k, name='top-k')
        tf.add_to_collection("top_k", top_k_test[1])
        target_loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_logits - neg_logits))) + 1e-24
        reg_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        target_loss += reg_losses
        loss =  target_loss

    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam2').minimize(loss, var_list=[softmax_var, weight])
    itemrec.save_impwei(mask_var,weight,taskID,case, extention, prune_percentage, reuse=True)  # save important weight

    unitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized_vars.append(var)

    initialize_op = tf.variables_initializer(unitialized_vars)
    sess.run(initialize_op)

    # Train the model and print metrices
    print("Training the model...")
    numIters = 1
    mrr5 = []
    hit5 = []
    ndcg5 = []
    accuracy = []
    iters = []
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            iters.append(numIters)
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            pos_batch = item_batch[:, -1]  # [3 6] used for negative sampling
            source_batch = item_batch[:, :-2]
            pos_target = item_batch[:, -1:]  # [[3][6]]
            neg_target = np.random.choice(targets_len_nozero, len(pos_batch), p=dl.prob)
            neg_target = np.array(neg_target + 1)
            neg_target = neg_target[:, np.newaxis]
            _, loss_out = sess.run( [optimizer, loss], feed_dict={itemrec.itemseq_input: item_batch, itemseq_input_target_pos: pos_target, itemseq_input_target_neg: neg_target})

            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------train1")
                print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss_out, iter, batch_no, numIters, train_set.shape[0] / batch_size))
               
                batch_no_test = 0
                batch_size_test = batch_size
                curr_preds_5=[]
                rec_preds_5=[]
                ndcg_preds_5=[]
                accuracy_pred=[]
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    pos_batch = item_batch[:, -1]
                    [top_k_batch] = sess.run([top_k_test], feed_dict={itemrec.itemseq_input: item_batch})
                    top_k = np.squeeze(top_k_batch[1])  # remove one dimension since e.g., [[[1,2,4]],[[34,2,4]]]-->[[1,2,4],[34,2,4]]
                    for i in range(top_k.shape[0]):
                        top_k_per_batch = top_k[i]
                        predictmap = {ch: i for i, ch in enumerate(top_k_per_batch)}  # add 2
                        true_item = pos_batch[i]
                        rank = predictmap.get(true_item)  # add 3
                        if rank ==None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)
                            ndcg_preds_5.append(0.0)
                            accuracy_pred.append(0.0)
                        else:
                            if rank == 1:
                                accuracy_pred.append(1.0)
                            else:
                                accuracy_pred.append(0.0)
                            MRR_5 = 1.0/(rank+1)
                            Rec_5=1.0
                            ndcg_5 = 1.0 / math.log(rank + 2, 2)
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)
                            ndcg_preds_5.append(ndcg_5)
                    batch_no_test += 1
                
                print("mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), 
                      "hit_5:", sum(rec_preds_5) / float(len(rec_preds_5)),   
                      "ndcg_5:", sum(ndcg_preds_5) / float(len(ndcg_preds_5)),
                      "accuracy:", sum(accuracy_pred) / float(len(accuracy_pred)))
                mrr5.append(sum(curr_preds_5) / float(len(curr_preds_5)))
                hit5.append(sum(rec_preds_5) / float(len(rec_preds_5)))
                ndcg5.append(sum(ndcg_preds_5) / float(len(ndcg_preds_5)))
                accuracy.append(sum(accuracy_pred) / float(len(accuracy_pred)))
                   
            batch_no += 1
            numIters += 1

    # Metrics evaluation
    metric_evaluation.Metric_Evaluation({'metric': 'MRR5', 'metric_values': mrr5, 'iters': iters, 'datapath': args.datapath, 'task': 'T3', 'mode': 'pretrain', 'path' : args.path})
    metric_evaluation.Metric_Evaluation({'metric': 'HIT5', 'metric_values': hit5, 'iters': iters, 'datapath': args.datapath, 'task': 'T3', 'mode': 'pretrain', 'path' : args.path})
    metric_evaluation.Metric_Evaluation({'metric': 'NDCG5', 'metric_values': ndcg5, 'iters': iters, 'datapath': args.datapath, 'task': 'T3', 'mode': 'pretrain', 'path' : args.path})
    metric_evaluation.Metric_Evaluation({'metric': 'Accuracy', 'metric_values': accuracy, 'iters': iters, 'datapath': args.datapath, 'task': 'T3', 'mode': 'pretrain', 'path' : args.path})

    # Save the masking for t3
    _mask_val_list = sess.run(itemrec.mask_val_list_task)
    for layer_id, dilation in enumerate(model_para['dilations']):
        resblock_type = "decoder"
        resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
        with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("dilated_conv1"):
                name_conv1 = "mask_filter/{}_mask_val".format(taskID)

                init_conv1 = tf.constant(_mask_val_list[2 * layer_id])
                mask_val_conv1 = tf.get_variable(name_conv1,
                                                 initializer=init_conv1, trainable=False)  # there is no optimizer
            with tf.variable_scope("dilated_conv2"):
                name_conv2 = "mask_filter/{}_mask_val".format(taskID)
                init_conv2 = tf.constant(_mask_val_list[2 * layer_id + 1])
                mask_val_conv2 = tf.get_variable(name_conv2,
                                                 initializer=init_conv2, trainable=False)

    unitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized_vars.append(var)
    initialize_op = tf.variables_initializer(unitialized_vars)
    sess.run(initialize_op)
    

    saver_ft = tf.train.Saver()
    save_path = saver_ft.save(sess, os.path.join(path, 'T3', 'pretrain') + "/model_nextitnet_transfer.ckpt".format(iter, numIters))
    print("Save models done!")


if __name__ == '__main__':
    main(sys.argv[4])
