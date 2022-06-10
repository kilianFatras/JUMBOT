import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network, my_loss
import lr_schedule, data_list
from utils import *
import ot 


def image_train(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    #hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    #hist_tar = hist_tar / hist_tar.sum()
    return accuracy#, hist_tar, mean_ent

def train(args):
    alpha = args.alpha
    lambda_t = args.lambda_t
    reg_m = args.reg_m
    print(alpha, lambda_t, reg_m)
    
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    
    source_labels = torch.zeros((len(dsets["source"])))

    for i, data in enumerate(dsets["source"]):
        source_labels[i] = data[1]

    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    dset_loaders["source"] = torch.utils.data.DataLoader(dsets["source"],
                                                         batch_sampler=train_batch_sampler, 
                                                         num_workers=args.worker)
    
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False,
                                        num_workers=args.worker)

    if "ResNet" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.ResNetFc(**params)
    
    if "VGG" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.VGGFc(**params)

    base_network = base_network.cuda()

    parameter_list = base_network.get_parameters()
    base_network = torch.nn.DataParallel(base_network).cuda() 

    ## set optimizer
    optimizer_config = {"type":torch.optim.SGD, "optim_params":
                        {'lr':args.lr, "momentum":0.9, "weight_decay":5e-4, "nesterov":True}, 
                        "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                    }
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    total_epochs = args.max_iterations // args.test_interval

    for i in range(args.max_iterations + 1):

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            # obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            
            best_model = base_network.state_dict()
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            args.out_file.write(log_str+"\n")
            args.out_file.flush()
            print(log_str)

        if i % args.test_interval == 0:
            log_str = "\n{}, iter: {:05d}, source/ target: {:02d} / {:02d}\n".format(args.name, i, train_bs, train_bs)
            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)
            
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        
        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs, ys = iter_source.next()
        xt, _ = iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        g_xs, f_g_xs = base_network(xs)
        g_xt, f_g_xt = base_network(xt)
        
        pred_xt = F.softmax(f_g_xt, 1)

        classifier_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys)

        ys = F.one_hot(ys, num_classes=args.class_num).float()

        M_embed = torch.cdist(g_xs, g_xt)**2
        M_sce = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
        M = alpha * M_embed + lambda_t * M_sce  # Ground cost

        #OT computation
        a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), 
                                                     0.01, reg_m=reg_m)
        pi = torch.from_numpy(pi).float().cuda()
        transfer_loss = 10. * torch.sum(pi * M)

        if i%100==0:
            print(torch.sum(pi), transfer_loss, torch.min(M), torch.min(M_embed), torch.min(M_sce))
        
        total_loss = classifier_loss + transfer_loss 
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))
    
    log_str = 'Acc: ' + str(np.round(temp_acc*100, 2)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return temp_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=65, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers") 
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])
    
    parser.add_argument('--dset', type=str, default='office_home', choices=["office", "office_home", "imagenet_caltech"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--alpha', type=float, default=1)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.max_iterations = 5000
        args.test_interval = 500
        args.lr=1e-3

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 2000
        args.test_interval = 200
        args.lr=1e-4

    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        args.class_num = 1000
        if args.s == 1:
            args.class_num = 256

        args.max_iterations = 40000
        args.test_interval = 4000
        args.lr=1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '_list.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join('ckp/partial', args.net, args.dset, args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    args.out_file.write(str(args)+'\n')
    args.out_file.flush()

    #train(args)
    list_alpha = [0.003]
    list_lambda_t = [0.75]
    list_reg_m = [0.06]
    results = []
    for alpha in list_alpha:
        for lambda_t in list_lambda_t:
            for reg_m in list_reg_m:
                args.alpha = alpha
                args.lambda_t = lambda_t
                args.reg_m = reg_m
                
                results.append(train(args))
                print(results)