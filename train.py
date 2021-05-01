import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel

from bart_model import BartModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from nlgeval import compute_metrics, getListMeteor

model_args = Seq2SeqArgs()
model_args.evaluate_during_training_steps = 1000000
model_args.num_train_epochs = 20
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.use_early_stopping = True
model_args.save_steps = 100000
model_args.top_k = 3
model_args.top_p = 0.95
model_args.max_length = 30
model_args.max_seq_length = 64
model_args.length_penalty = 1.2
model_args.save_model_every_epoch = True
model_args.train_batch_size = 32
model_args.eval_batch_size = 32
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.save_optimizer_and_scheduler = False
model_args.output_dir = 'Bart/'
model_args.best_model_dir = 'Bart/best_model'
model_args.overwrite_output_dir = True
model_args.early_stopping_metric = 'Meteor'
model_args.early_stopping_metric_minimize = False

# 模型初始化，进行fine-tune
model = BartModel(pretrained_model=None,args=model_args, model_config='config.json', vocab_file='vocab')

train_df = pd.read_csv('data/train.csv')
eval_df = pd.read_csv('data/eval.csv')
test_df = pd.read_csv('data/test.csv')

train_df.columns = ['input_text', 'target_text']
eval_df.columns = ['input_text', 'target_text']
test_df.columns = ['input_text', 'target_text']

def Meteor(labels, preds):
    score = getListMeteor(preds, labels)
    return score

# 训练
model.train_model(train_df, eval_data=eval_df, Meteor=Meteor)


# 加载本地训练好的模型
model = BartModel(pretrained_model='Bart/best_model',args=model_args, model_config='Bart/best_model/config.json', vocab_file='Bart/best_model')

# 测试
test_list = test_df['input_text'].tolist()
pred_list = model.predict(
        test_list
    )
true_list = test_df['target_text'].tolist()

column_name = ['title']
nl_df = pd.DataFrame(true_list, columns=column_name)
nl_df.to_csv('result/code_true_bart.csv', index=None, header=False)
nl_df = pd.DataFrame(pred_list, columns=column_name)
nl_df.to_csv('result/code_pred_bart.csv', index=None, header=False)



compute_metrics(hypothesis='result/code_pred_bart.csv',
                                   references=['result/code_true_bart.csv'],no_glove=True,no_skipthoughts=True)
