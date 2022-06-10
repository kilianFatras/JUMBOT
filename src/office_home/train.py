#inspired from https://github.com/thuml/CDAN and https://github.com/ZJULearning/ALDA
import argparse
import os
import os.path as osp

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList, ImageList_label
from torch.autograd import Variable
import random
import math
import torch.nn.functional as F
from utils import *
import ot

import wandb
import os

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    dataset = loader['test']
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(dataset[i]) for i in range(10)]
            for i in range(len(dataset[0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    feature, predict_out = model(inputs[j])
                    predict_out = nn.Softmax(dim=1)(predict_out)
                    outputs.append(predict_out)
                outputs = sum(outputs) / 10
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(dataset)
            for i in range(len(dataset)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feature, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(config):

    exp_tag = config["exp_tag"]
    print(f'{config["exp_full_tag"]} is started')

    criterion=nn.CrossEntropyLoss()
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    source_list = ['.'+i for i in open(data_config["source"]["list_path"]).readlines()]
    target_list = ['.'+i for i in open(data_config["target"]["list_path"]).readlines()]

    dsets["source"] = ImageList(source_list, \
                                transform=prep_dict["source"])
    
    source_labels = torch.zeros((len(dsets["source"])))

    for i, data in enumerate(dsets["source"]):
        source_labels[i] = data[1]

    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    dset_loaders["source"] = torch.utils.data.DataLoader(dsets["source"], batch_sampler=train_batch_sampler, num_workers=config['args'].num_worker)

    dsets["target"] = ImageList(target_list, \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=config['args'].num_worker, drop_last=True)
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
            dsets["test"] = [ImageList(test_list, \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=config['args'].num_worker) for dset in dsets['test']]
    else:
        test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
        dsets["test"] = ImageList(test_list, \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=config['args'].num_worker)
    
    dsets["target_label"] = ImageList_label(target_list, \
                            transform=prep_dict["target"])
    dset_loaders["target_label"] = DataLoader(dsets["target_label"], batch_size=test_bs, \
            shuffle=False, num_workers=config['args'].num_worker, drop_last=False)

    class_num = config["network"]["params"]["class_num"]
    alpha = config["alpha"]
    lambda_t = config["lambda_t"]
    reg_m = config["reg_m"]
    print(alpha, lambda_t, reg_m)
    

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    # if config["restore_path"]:
    #     checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))["base_network"]
    #     ckp = {}
    #     for k, v in checkpoint.items():
    #         if "module" in k:
    #             ckp[k.split("module.")[-1]] = v
    #         else:
    #             ckp[k] = v
    #     base_network.load_state_dict(ckp)
    #     log_str = "successfully restore from {}".format(osp.join(config["restore_path"], "best_model.pth"))
    #     config["out_file"].write(log_str+"\n")
    #     config["out_file"].flush()
    #     print(log_str)

    parameter_list = base_network.get_parameters()
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []

    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    if config['num_gpu'] > 1:
        print("DataParallel")
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(config['num_gpu'])])
        
    loss_params = config["loss"]
    high = loss_params["trade_off"]
    begin_label = False
    # writer = SummaryWriter(config["output_path"])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = 0
    classifier_loss_value = 0
    total_loss_value = 0
    best_acc = 0
    loss_value = 0

    for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"]-1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = base_network #nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_step = i
                best_acc = temp_acc

                wandb.log({
                    f'{exp_tag}_iteration': i,
                    f'{exp_tag}_precision': temp_acc,
                    f'{exp_tag}_classifier_loss': classifier_loss_value,
                    f'{exp_tag}_transfer_loss': transfer_loss_value,
                    f'{exp_tag}_total_loss': total_loss_value,
                })

            print("class_loss: {:.3f}".format(loss_value))
            loss_value = 0

            #show val result on tensorboard
            # images_inv = prep.inv_preprocess(xs.clone().cpu(), 3)
            # for index, img in enumerate(images_inv):
                # writer.add_image(str(index)+'/Images', img, i)
            
        if i == config["num_iterations"]-1:
            print('end of training')

            wandb.log({
                'exp_full_tag': config["exp_full_tag"],
                f'{exp_tag}_best_step': best_step,
                f'{exp_tag}_best_acc': best_acc,
            })
            break

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        xs, ys = iter_source.next()  # source minibatch
        xt, _ = iter_target.next()  # target minibatch
        xs, xt, ys = Variable(xs).cuda(), Variable(xt).cuda(), Variable(ys).cuda()
        
        if config['method'] == 'JUmbOT':

            g_xs, f_g_xs = base_network(xs)  # source embedded data
            g_xt, f_g_xt = base_network(xt)  # target embedded data
            pred_xt = F.softmax(f_g_xt,1)

            classifier_loss = criterion(f_g_xs, ys)

            ys = F.one_hot(ys, num_classes=class_num).float()  # Get One Hot probability vectors

            M_embed = torch.cdist(g_xs, g_xt)**2  # Term on embedded data 
            M_sce = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))  # Term on labels
            M = alpha * M_embed + lambda_t * M_sce  # Ground cost

        else:
            method_name = config['method']
            raise ValueError(f'Method {method_name} has not been implemented.')

        #OT computation
        a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), 
                                                     0.01, reg_m=reg_m)  
        pi = torch.from_numpy(pi).float().cuda()  # Transport plan between minibatches
        transfer_loss = torch.sum(pi * M)
        if i%100==0:
            print(torch.sum(pi), transfer_loss, torch.min(M), torch.min(M_embed), torch.min(M_sce))
        
        loss_value += classifier_loss.item() / config["test_interval"]           
        total_loss =  classifier_loss + transfer_loss

        classifier_loss_value += classifier_loss.item() / config["test_interval"]   
        transfer_loss_value += transfer_loss.item() / config["test_interval"]
        total_loss_value = classifier_loss_value + transfer_loss_value

        total_loss.backward()
        optimizer.step()

    # checkpoint = {"base_network": temp_model.state_dict()}
    # torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    return best_acc

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='UOT')
    parser.add_argument('--num_gpu', type=int, default='2', help="num of gpus")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office_home'], help="The dataset or source dataset used")
    # parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_31_list.txt', help="The source dataset path list")
    # parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_10_list.txt', help="The target dataset path list")
    # parser.add_argument('--s_dset_id', type=int, default=0, help="The source dataset path list, e.g. if you set 0 -> office_home/Art.txt")
    # parser.add_argument('--t_dset_id', type=int, default=1, help="The target dataset path list, e.g. if you set 1 -> office_home/Clipart.txt")

    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")

    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="trade off between supervised loss and self-training loss")
    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--cos_dist', type=str2bool, default=False, help="the classifier uses cosine similarity.")

    parser.add_argument('--weight_type', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--test_10crop', type=str2bool, default=True)

    # parser.add_argument('--adv_weight', type=float, default=1.0, help="weight of adversarial loss")
    # parser.add_argument('--source_detach', default=False, type=str2bool, help="detach source feature from the adversarial learning")

    parser.add_argument("--wandb_entity", type=str, default='rlopt', help="entitiy of wandb team")
    parser.add_argument("--wandb_project_name", type=str, default='default_project', help="entitiy of wandb project")
    parser.add_argument('--wandb_offline', action = 'store_true')
    parser.add_argument('--debug_mode', action = 'store_true')

    args = parser.parse_args()

    # ============= GPU Device =============
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.num_gpu > 0) else "cpu")
    print(f'device : {device}')

    # ============= set seed =============
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=True

    # ============= train config =============
    config = {}
    config['args'] = args
    config['method'] = args.method
    # config["gpu"] = args.gpu_id
    config["num_gpu"] = args.num_gpu
    # config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True


    if config['num_gpu'] > 1:
        args.batch_size = args.batch_size
        print(f"gpus:{list( range( config['num_gpu'])) }, batch size:{args.batch_size}")

    config["prep"] = {"test_10crop":args.test_10crop, 'params':{"resize_size":256, "crop_size":224}}
    config["loss"] = {"trade_off":args.trade_off}

    # ================ Model Arch ================
    if "ResNet" in args.net:
        net = network.ResNetFc
        if args.dset == "office-home":
            config["network"] = {"name":net, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":512,
                          "new_cls":True, "cos_dist":args.cos_dist} }
        elif args.dset == "visda":
            config["network"] = {"name":net, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256,
                          "new_cls":True, "cos_dist":args.cos_dist} }
            
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    # ================ Optimizer ================
    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }


    # ================ Dataset Dependent Configuration ================ 
    config["dataset"] = args.dset
    if config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
        domain_names = ['Art', 'Clipart', 'Product', 'Real_World']

    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        domain_names = ['train_list', 'validation_list', 'test_list']
        #config['loss']["trade_off"] = 1.0
    else:
        raise ValueError('Dataset has not been implemented.')

    # ================ Data Loader ================ 

    if os.environ['CLUSTER_NAME'] == 'tokyotech_cluster':
        data_folder = os.environ['HINADORI_LOCAL_SCRATCH'] # For Tokyo Tech Cluster
    elif os.environ['CLUSTER_NAME'] == 'mila_cluster':
        data_folder = os.environ['SLURM_TMPDIR'] # For Mila Cluster   
    elif os.environ['CLUSTER_NAME'] == 'graham':
        data_folder = os.environ['SLURM_TMPDIR'] # For Graham Cluster   
    else:
        cluster_name = os.environ['CLUSTER_NAME']
        raise ValueError(f'Dataset {cluster_name} has not been supported.')
    
    #OH param : alpha = 0.01, lambda = 0.5, reg_m = 0.5
    #VisDA param : alpha = 0.005, lambda = 1., reg_m = 0.3
    
    # ================ Hyperparams ================ 
    results = []
    config["alpha"] = 0.01
    config["lambda_t"] = 0.5
    config["reg_m"] = 0.5

    # ================ Wandb ================ 
    wandb_exp_name = f'{args.method}_seed_{args.seed}'
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(config=args,
                project=args.wandb_project_name,
                name=wandb_exp_name,
                entity=args.wandb_entity)

    print(f'wandb_project_name: f{args.wandb_project_name}')
    print(f'wandb_exp_name: f{wandb_exp_name}')

    ####### Main
    counter = 0
    for count_s, domain_name_s in enumerate(domain_names):
        for count_t, domain_name_t in enumerate(domain_names):

            if count_s != count_t:

                s_dset_path = os.path.join(f'{data_folder}/', args.dset, f'{domain_names[count_s]}.txt')
                t_dset_path = os.path.join(f'{data_folder}/', args.dset, f'{domain_names[count_t]}.txt')

                exp_full_tag = f'{counter}:{domain_names[count_s]}->{domain_names[count_t]}'
                config["exp_tag"] = counter
                config["exp_full_tag"] = exp_full_tag

                config["data"] = {"source":{"list_path":s_dset_path, "batch_size":args.batch_size}, \
                                    "target":{"list_path":t_dset_path, "batch_size":args.batch_size}, \
                                    "test":{"list_path":t_dset_path, "batch_size":4}
                                    }

                print(f'counter: {counter}')
                print('s dataset: ', s_dset_path)
                print('t dataset: ', t_dset_path)

                train(config)


