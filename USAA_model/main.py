import torch
from USAA_model_1 import USSA_model

import argparse
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(6)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='weibo', choices=['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=1001, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=220, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--input_dim_G', type=int, default=768, help='Number of feature in X')
    parser.add_argument('--channels', type=int, default=6, help='Number of time sequence in X')
    parser.add_argument('--num_nodes', type=int, default=50, help='num of node in X')
    parser.add_argument('--num_heads', type=list, default=[8,8], help='num_heads')
    parser.add_argument('--h_DUGAT', type=list, default=[400, 200], help='Number of D node on hidden layer')
    parser.add_argument('--h_op', type=int, default=200, help='Number of dim after op_layer')
    parser.add_argument('--h_UDGAT', type=list, default=[200, 100], help='Number of D node on hidden layer')
    parser.add_argument('--hidden_LSTM', type=int, default= 100, help='Number of dimension on hidden layer')
    parser.add_argument('--dense_C', type=list, default=[100,50], help='Number of D node on dense layer')
    parser.add_argument('--lr', type=tuple, default=1e-4, help='lr_C')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\USAT-RD\result', help='Directory name to save the GAN_model')
    parser.add_argument('--model_dir', type=str, default=r'classifier', help='Directory name to save the GAN')
    parser.add_argument('--data', default=r'E:\WDK_workshop\USAT-RD\data\weibo_all_emb', type=str)
    parser.add_argument('--ispath', default=True, type=bool)
    parser.add_argument('--data_eval', default='', type=str)


    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")


    print(device)

    #model = classifier_model(args, device)
    model = USSA_model(args, device)


    if not os.path.exists(os.path.join(args.save_dir, args.model_dir)):
        os.mkdir(os.path.join(args.save_dir,  args.model_dir))
    if not os.path.exists(os.path.join(args.save_dir, args.model_dir, args.dataset)):
        os.mkdir(os.path.join(args.save_dir,args.model_dir, args.dataset))

    if os.path.exists(os.path.join(args.save_dir, args.model_dir, args.dataset, str(args.start_epoch) + '_classifier.pkl')):
        start_epoch = model.load(start_epoch=args.start_epoch)
        print("load epoch {} success!".format(start_epoch))
    else:
        start_epoch = 0
        print("start from epoch {}".format(start_epoch))

    #model.train(os.path.join(args.save_dir, args.data), start_epoch)
    model.train(args.data, start_epoch, args.ispath)
    print(" [*] Training finished!")







