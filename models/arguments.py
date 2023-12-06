import argparse
parser = argparse.ArgumentParser(description='Process some Parameters.')
parser.add_argument('--running_type', type=str, default="train",
                    help='running type, like: train, evaluate, test ...')
parser.add_argument('--diversity_topk', type=int, default=1000,
                    help='diversity_topk for finetune sample method.')
parser.add_argument('--model_alias', type=str, default="",
                    help='model alias to distinguish it from other models.')
parser.add_argument('--threshold', type=float, default=0.95,
                    help='threshold for the silver label extraction')
parser.add_argument('--update_method', type=str, default="average",
                    help='update method for silver label, now either average or finetune.')
parser.add_argument('--sample_method', type=str, default="all",
                    help='sample method for the self-training, now either finetune or all.')
parser.add_argument('--negative_sample_method', type=str, default="random",
                    help='sample method for the negative samples, now either probability or random.')
parser.add_argument('--model_type', type=str, default="cat_1",
                    help='type of the model.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate of the model.')
parser.add_argument('--EPOCHS', type=int, default=200,
                    help='Max train iteration number for each epoch.')
parser.add_argument('--ITERATIONS', type=int, default=15,
                    help='total iteration number for the self-training.')
parser.add_argument('--BATCH_SIZE', type=int, default=20000,
                    help='batch size.')
parser.add_argument('--encode_type', type=str, default="long_sent",
                    help='encode_type.')
parser.add_argument('--optimizer', type=str, default="Adadelta",
                    help='current support optimizers are: Adam, Adadelta.')
parser.add_argument('--use_model', type=str, default="pre-train",
                    help='current use model are: pre-train, lstm.')
parser.add_argument('--code_start_time', type=str, default="None",
                    help='if None then use current time, else should provide a time.')
parser.add_argument('--ebc_type', type=str, default="ebc",
                    help='ebc type, now either ebc or diva.')
parser.add_argument('--positive_partition',type=float,default=0.32,
                    help='positive partition-pai_p')
parser.add_argument('--gamma_p',type=int,default=8,
                    help='for focal loss --gamma for positive')
parser.add_argument('--gamma_n',type=int,default=5,
                    help='for focal loss --gamma for negative')
parser.add_argument('--alpha',type=float,default=0.5,
                    help='for focal loss --alpha')
parser.add_argument('--confident_ratio',type=float,default=0.07,help='confident ratio - for self paced learning')

# tree stages for epoch num
parser.add_argument('--warm_up_stages', type=int, default=20, help='warm up stages (0-20)')
parser.add_argument('--self_paced_stages', type=int, default=50, help='self paced stages (21-50)')
# parser.add_argument('--classification_layer_input_dim', type=int, default=768*4,
#                     help='the dimension for the input of the classification layer.')

# ns
parser.add_argument('--ns_base_loss', type=str, default='sigmoid', help='negative sample base loss func -sigmoid / bcsloss')
parser.add_argument('--ns_alpha_weight', type=int, default=40, help='negative sample alpha val --for popularity calculation')
parser.add_argument('--ns_loss_component', type=str, default='with_hard_negative_loss', help='whether to use hard negative loss')

parser.add_argument('--label_class', type=str, default='golden', help='golden or expert')
parser.add_argument('--diva_label_class',type=str,default='golden',help='golden or expert')

parser.add_argument('--file_version',type=str,default='none')
parser.add_argument('--uncertainty_limit',type=bool,default=True)
parser.add_argument('--tf_idf_type',type=str,default='old',help='new(tf-idf++)-site4 or old site10')

parser.add_argument('--with_valid',type=bool,default=True)
parser.add_argument('--light',type=bool,default=False,help='whether to use light version of the iteration model')
parser.add_argument('--clusters_num',type=int,default=15)
parser.add_argument('--clustering_times',type=int,default=10)
parser.add_argument('--PE_thres',type=float,default=0.5)
parser.add_argument('--ebc_thres',type=float,default=0.95)
parser.add_argument('--start_iter',type=int,default=1)
parser.add_argument('--easy_self_training',type=bool,default=False)
parser.add_argument('--joint_score_thres',type=float,default=0.01)
parser.add_argument('--debug',type=bool,default=False)
parser.add_argument('--discard_positive',type=bool,default=True)
parser.add_argument('--update_thres',type=float,default=0.03)
parser.add_argument('--dumping_weight',type=float,default=0.5)


parser.add_argument('--with_tf_idf',type=bool,default=True)
parser.add_argument('--with_novelty',type=bool,default=True)
parser.add_argument('--with_valid_semantic',type=bool,default=True)
parser.add_argument('--with_valid_disc',type=bool,default=True)
parser.add_argument('--only_joint',type=bool,default=False)
parser.add_argument('--valid_thres',type=float,default=0.1)

args = parser.parse_args()

