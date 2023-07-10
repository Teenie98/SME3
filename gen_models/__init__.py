import os
import argparse
import torch
from gen_models.MetaEmb import MetaEmb
from gen_models.SME import SME
from gen_models.SME2 import SME2
from gen_models.SME3 import SME3
from gen_models.GME import GME
from gen_models.MWUF import MetaScaling, MetaShifting, MWUF
from rec_models import device, set_seed, model_dict as rec_model_dict

model_dict = {'MetaEmb': MetaEmb, 'MWUF': MWUF, 'GME': GME, 'SME': SME, 'SME2': SME2, 'SME3': SME3}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SME3', help='MetaEmb MWUF GME SME SME2 SME3')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--meta_learning_rate', type=float, default=1e-3)
    parser.add_argument('--cold_learning_rate', type=float, default=1e-4)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_layer_size', type=int, default=64)
    parser.add_argument('--warm_up_batch_size', type=int, default=400)
    parser.add_argument('--warm_up_learning_rate', type=float, default=1e-3)
    parser.add_argument('--generator_hidden_size', type=int, default=16)
    parser.add_argument('--generator_train_batch_size', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--base_model', type=str, default='afm', help='deepfm wideanddeep ipnn opnn afm')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    set_seed(args.seed)
    assert args.model in model_dict
    assert args.base_model in rec_model_dict

    print('loading model {}...'.format(args.base_model))
    rec_model = rec_model_dict[args.base_model](args).to(device)
    rec_model.load_state_dict(torch.load("../rec_models/save_p/{}_parameter.pkl".format(args.base_model)))

    test_auc, test_logloss = rec_model.predict()
    print('base model test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

    model = model_dict[args.model](args).to(device)
    if args.model == "MWUF":
        print('meta network training...')
        scaling_net = MetaScaling(args).to(device)
        shifting_net = MetaShifting(args).to(device)
        model.meta_network_train(scaling_net, shifting_net, rec_model)

        test_auc, test_logloss = rec_model.predict()
        print('init\ntest auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

        print('warm up training...')
        model.warm_up_train(rec_model, scaling_net, shifting_net, args.warm_up_batch_size, args.warm_up_learning_rate, 'MovieID')
    else:
        print('generator training...')
        model.generate_train(rec_model,
                             args.generator_train_batch_size,
                             args.meta_learning_rate,
                             args.cold_learning_rate,
                             args.alpha)

        model.init_id_embedding(rec_model)
        test_auc, test_logloss = rec_model.predict()
        print('init\ntest auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

        print('warm up training...')
        rec_model.warm_up_train(args.warm_up_batch_size, args.warm_up_learning_rate, 'MovieID')
