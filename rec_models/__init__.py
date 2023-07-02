from rec_models.afm import AFM
from rec_models.deepfm import DeepFM
from rec_models.ipnn import IPNN
from rec_models.opnn import OPNN
from rec_models.wideanddeep import WideAndDeep
from rec_models.base import device, set_seed

model_dict = {'wideanddeep': WideAndDeep,
              'deepfm': DeepFM,
              'ipnn': IPNN,
              'opnn': OPNN,
              'afm': AFM}

if __name__ == "__main__":
    import torch
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=2023)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--embedding_size', type=int, default=16)
        parser.add_argument('--hidden_layer_size', type=int, default=64)
        parser.add_argument('--warm_up_batch_size', type=int, default=400)
        parser.add_argument('--warm_up_learning_rate', type=float, default=1e-3)
        parser.add_argument('--pretrain', type=int, default=2)
        parser.add_argument('--model', type=str, default='deepfm', help="deepfm wideanddeep ipnn opnn afm")
        args = parser.parse_args()
        return args

    args = parse_args()
    set_seed(args.seed)
    assert args.model in model_dict
    model = model_dict[args.model](args).to(device)

    if args.pretrain == 2:
        print('cross domain dataset: training model {}...'.format(args.model))
        model.pre_train(args.batch_size, args.learning_rate, '../data/dataset_sys1')
        torch.save(model.state_dict(), "./cross_domain_p/{}_parameter.pkl".format(args.model))
    elif args.pretrain == 1:
        print('training model {}...'.format(args.model))
        model.pre_train(args.batch_size, args.learning_rate)
        torch.save(model.state_dict(), "./save_p/{}_parameter.pkl".format(args.model))
    else:
        print('load model {}...'.format(args.model))
        model.load_state_dict(torch.load("./save_p/{}_parameter.pkl".format(args.model)))

    test_auc, test_logloss = model.predict()
    print('test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

    print('warm up training...')
    model.warm_up_train(args.warm_up_batch_size, args.warm_up_learning_rate, 'MovieID')
