import argparse

import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch.autograd import Variable
import torch_geometric.transforms as T
from model import AutoLink_l3Table, SearchGraph_l31
from utils import IndexLoader
from linkmodel import PygLinkPropPredDataset, Evaluator
import numpy as np
from logger import Logger
import time

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(vector, dalpha_, model, vnet, adj_t, train_edge, train_edge_neg, r=1e-2):
    all_vector = vector + dalpha_
    R = r / _concat(all_vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.add_(R, v)

    # graph_representations_v_model, output_v_model = model(data)
    h = model(adj_t)
    loss = model.compute_loss(h, vnet, train_edge, train_edge_neg)

    variable_list = [param for param in vnet.parameters()]
    grads_p = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.sub_(2 * R, v)

    # graph_representations_v_model, output_v_model = model(data)
    h = model(adj_t)
    loss = model.compute_loss(h, vnet, train_edge, train_edge_neg)

    grads_n = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def train(model, v_model, arch_net, adj_t, split_edge, optimizer, batch_size, num_nodes, device):
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    v_model.eval()
    arch_net.eval()

    pos_train_edge = split_edge['train']['edge'].to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        train_edge = pos_train_edge[perm].t()
        train_edge_neg = negative_sampling(edge_index, num_nodes=num_nodes,
                                 num_neg_samples=perm.size(0), method='dense')
        with torch.no_grad():
            h = v_model(adj_t)
            pos_arch, neg_arch = v_model.compute_arch(h, arch_net, train_edge, train_edge_neg)

        h = model(adj_t)
        loss = model.compute_loss_arch(h, pos_arch, neg_arch, train_edge, train_edge_neg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        num_examples = perm.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, arch_net, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    arch_net.eval()

    h = model(adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [model.compute_pred(h, arch_net, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [model.compute_pred(h, arch_net, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [model.compute_pred(h, arch_net, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [model.compute_pred(h, arch_net, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [model.compute_pred(h, arch_net, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


@torch.no_grad()
def test_model(model, v_model, arch_net, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    v_model.eval()
    arch_net.eval()

    h = model(adj_t)
    h_v = v_model(adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        atten_arch = v_model.compute_arch_input(h_v, arch_net, edge)
        pos_train_preds += [model.compute_pred_arch(h, atten_arch, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        atten_arch = v_model.compute_arch_input(h_v, arch_net, edge)
        pos_valid_preds += [model.compute_pred_arch(h, atten_arch, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        atten_arch = v_model.compute_arch_input(h_v, arch_net, edge)
        neg_valid_preds += [model.compute_pred_arch(h, atten_arch, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        atten_arch = v_model.compute_arch_input(h_v, arch_net, edge)
        pos_test_preds += [model.compute_pred_arch(h, atten_arch, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        atten_arch = v_model.compute_arch_input(h_v, arch_net, edge)
        neg_test_preds += [model.compute_pred_arch(h, atten_arch, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def train_arch(split_edge, adj_t, batch_size, args, num_nodes, evaluator, save_path_model, save_path_arch, device):

    model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.use_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    v_model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.use_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    arch_net = SearchGraph_l31(args.hidden_channels, args.arch_dim, args.arch_layers, cat_type=args.cat_type, temperature=args.temperature).to(device)
    v_arch_net = SearchGraph_l31(args.hidden_channels, args.arch_dim, args.arch_layers, cat_type=args.cat_type, temperature=args.temperature).to(device)
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    metric = 'Hits@20'

    model.reset_parameters()
    arch_net.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr)

    v_optimizer = torch.optim.Adam(
        list(v_model.parameters()) + list(v_arch_net.parameters()),
        lr=args.lr)

    optimizer_arch = torch.optim.Adam(arch_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate_min)

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    # pos_valid_edge_neg = split_edge['valid']['edge_neg'].to(device)
    pos_valid_edge_neg = negative_sampling(edge_index, num_nodes=num_nodes,
                                       num_neg_samples=pos_valid_edge.size(0), method='dense').to(device).t()
    val_loader = IndexLoader(pos_valid_edge.size(0), batch_size, shuffle=True, drop_last=False)

    best_valid = 0.0
    best_epoch = 0
    cnt_wait = 0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[-1]
        steps = 0
        epoch_loss = 0.0
        model.train()
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            arch_net.train()
            v_model.load_state_dict(model.state_dict())
            v_arch_net.load_state_dict(arch_net.state_dict())
            train_edge = pos_train_edge[perm].t()
            train_edge_neg = negative_sampling(edge_index, num_nodes=num_nodes,
                                               num_neg_samples=perm.size(0), method='dense')
            h = v_model(adj_t)
            loss = v_model.compute_loss(h, v_arch_net, train_edge, train_edge_neg)

            v_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(v_arch_net.parameters(), 1.0)
            v_optimizer.step()

            # for valid L_valid
            valid_perm = val_loader.next()
            valid_edge = pos_valid_edge[valid_perm].t()
            valid_edge_neg = pos_valid_edge_neg[valid_perm].t()
            h = v_model(adj_t)
            loss = v_model.compute_loss(h, v_arch_net, valid_edge, valid_edge_neg)

            v_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(v_arch_net.parameters(), 1.0)

            dalpha = [v.grad for v in v_arch_net.parameters()]
            dalpha_ = [v.grad.data for v in v_arch_net.parameters()]

            vector = [v.grad.data for v in v_model.parameters()]
            # w^{+-}=w+- \lambda * Gradient_w^*L_val(w^*, \alpha)
            implicit_grads = _hessian_vector_product(vector, dalpha_, model, arch_net, adj_t, train_edge, train_edge_neg)
            # Update parameters for architecture based on Gradient_\alpha L_val(w^*, \alpha) - 2-th term
            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(lr, ig.data)

            i = 0
            for name, params in arch_net.named_parameters():
                if params.requires_grad:
                    if params.grad is None:
                        params.grad = Variable(dalpha[i].data)
                    else:
                        params.grad.data.copy_(dalpha[i].data)
                    i += 1
            torch.nn.utils.clip_grad_norm_(arch_net.parameters(), 1.0)
            optimizer_arch.step()

            # update model
            h = model(adj_t)
            arch_net.eval()
            loss = model.compute_loss(h, arch_net, train_edge, train_edge_neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss.item()))
            steps += 1

        results = test(model, arch_net, adj_t, split_edge, evaluator, args.batch_size)
        valid_hits = results[metric][1]
        if valid_hits > best_valid:
            best_valid = valid_hits
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_model)
            torch.save(arch_net.state_dict(), save_path_arch)
        else:
            cnt_wait += 1

        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print("--------" + key)
            print(
                  f'Epoch: {epoch:02d} / {args.epochs + 1:02d}, '
                  f'Best_epoch: {best_epoch:02d}, '
                  f'Best_valid: {100 * best_valid:.2f}%, '
                  f'Loss: {epoch_loss / steps:.4f}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
    print('*******Architecture Search Finished********')


def subgraph_count(split_edge, batch_size, v_model, arch_net, adj_t, num_layer, device, infe_set='train'):
    pos_train_edge = split_edge[infe_set]['edge'].to(device)
    v_model.eval()
    arch_net.eval()
    sub_count = np.zeros(shape=(num_layer * num_layer, ), dtype=float)
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=False):
        train_edge = pos_train_edge[perm].t()
        with torch.no_grad():
            h = v_model(adj_t)
            pos_arch, neg_arch = v_model.compute_arch(h, arch_net, train_edge, train_edge)
        count_pos = pos_arch.sum(0).data.cpu().numpy()
        sub_count += count_pos
    return sub_count.reshape(num_layer, num_layer)


def train_model(adj_t, split_edge, batch_size, args, evaluator, save_path_model_test, save_path_model, save_path_arch, num_nodes, device):
    model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.test_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    v_model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.use_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    arch_net = SearchGraph_l31(args.hidden_channels, args.arch_dim, args.arch_layers, cat_type=args.cat_type, temperature=args.temperature).to(device)

    metric = 'Hits@20'

    model.reset_parameters()
    v_model.load_state_dict(torch.load(save_path_model))
    arch_net.load_state_dict(torch.load(save_path_arch))

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr)

    best_valid = 0.0
    best_epoch = 0
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, v_model, arch_net, adj_t, split_edge, optimizer,
                     args.batch_size, num_nodes, device)
        results = test_model(model, v_model, arch_net, adj_t, split_edge, evaluator,
                       args.batch_size)

        valid_hits = results[metric][1]
        if valid_hits > best_valid:
            best_valid = valid_hits
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_model_test)

        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(
                  f'Epoch: {epoch:02d} / {args.epochs + 1:02d}, '
                  f'Best_epoch: {best_epoch:02d}, '
                  f'Best_valid: {100 * best_valid:.2f}%, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')
        print('***************')

    model.load_state_dict(torch.load(save_path_model_test))
    results = test_model(model, v_model, arch_net, adj_t, split_edge, evaluator, batch_size)

    sub_count_train = subgraph_count(split_edge, batch_size, v_model, arch_net, adj_t, args.num_layers, device)
    sub_count_test = subgraph_count(split_edge, batch_size, v_model, arch_net, adj_t, args.num_layers, device, infe_set='test')
    return results, sub_count_train, sub_count_test


def train_fine(adj_t, split_edge, batch_size, args, evaluator, save_path_model_test, save_path_model, save_path_arch, num_nodes, device):
    model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.test_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    v_model = AutoLink_l3Table(args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout,
                             args.use_sage, num_nodes, lin_layers=args.lin_layers, cat_type=args.cat_type).to(device)
    arch_net = SearchGraph_l31(args.hidden_channels, args.arch_dim, args.arch_layers, cat_type=args.cat_type, temperature=args.temperature).to(device)

    metric = 'Hits@20'

    model.load_state_dict(torch.load(save_path_model))
    v_model.load_state_dict(torch.load(save_path_model))
    arch_net.load_state_dict(torch.load(save_path_arch))

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr_fine)

    best_valid = 0.0
    best_epoch = 0
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, v_model, arch_net, adj_t, split_edge, optimizer,
                     args.batch_size, num_nodes, device)
        results = test_model(model, v_model, arch_net, adj_t, split_edge, evaluator,
                       args.batch_size)

        valid_hits = results[metric][1]
        if valid_hits > best_valid:
            best_valid = valid_hits
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_model_test)

        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(
                  f'Epoch: {epoch:02d} / {args.epochs + 1:02d}, '
                  f'Best_epoch: {best_epoch:02d}, '
                  f'Best_valid: {100 * best_valid:.2f}%, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')
        print('***************')

    model.load_state_dict(torch.load(save_path_model_test))
    results = test_model(model, v_model, arch_net, adj_t, split_edge, evaluator, batch_size)

    # sub_count_train = subgraph_count(split_edge, batch_size, v_model, arch_net, adj_t, args.num_layers)
    # sub_count_test = subgraph_count(split_edge, batch_size, v_model, arch_net, adj_t, args.num_layers, infe_set='test')
    sub_count_train = 0
    sub_count_test = 0
    return results, sub_count_train, sub_count_test


def main(num_nodes, adj_t, split_edge, args, device):

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    loggers_fine = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    sub_count_train = 0
    sub_count_test = 0
    for run in range(args.runs):
        t1 = time.time()
        save_path_model = 'weight/l3bl-' + args.use_sage + '_{}'.format(args.test_sage) + '_{}'.format(args.dataset) \
                          + '_arch-{}-{}'.format(args.arch_layers, args.arch_dim) + '_{}'.format(args.num_layers) + "-lin{}".format(args.lin_layers) \
                          + '_hidd{}'.format(args.hidden_channels) + '_{}'.format(
            args.cat_type) + '_model_{}.pth'.format(run)
        save_path_model_test = 'weight/l3bl-' + args.test_sage + '_{}'.format(args.test_sage) + '_{}'.format(
            args.dataset) \
                               + '_arch-{}-{}'.format(args.arch_layers, args.arch_dim) + '_{}'.format(args.num_layers) + "-lin{}".format(args.lin_layers) \
                               + '_hidd{}'.format(args.hidden_channels) + '_{}'.format(
            args.cat_type) + '_modeltest_{}.pth'.format(run)

        save_path_model_fine = 'weight/l3bl-' + args.test_sage + '_{}'.format(args.test_sage) + '_{}'.format(
            args.dataset) \
                               + '_arch-{}-{}'.format(args.arch_layers, args.arch_dim) + '_{}'.format(args.num_layers) + "-lin{}".format(args.lin_layers) \
                               + '_hidd{}'.format(args.hidden_channels) + '_{}'.format(
            args.cat_type) + '_modelfine_{}.pth'.format(run)

        save_path_arch = 'weight/l3bl-' + args.use_sage + '_{}'.format(args.test_sage) + '_{}'.format(args.dataset) \
                         + '_arch-{}-{}'.format(args.arch_layers, args.arch_dim) + '_{}'.format(args.num_layers) + "-lin{}".format(args.lin_layers) \
                         + '_hidd{}'.format(args.hidden_channels) + '_{}'.format(args.cat_type) + '_arch_{}.pth'.format(
            run)

        train_arch(split_edge, adj_t, args.batch_size, args, num_nodes, evaluator, save_path_model, save_path_arch, device)
        t2 = time.time()
        results_fine, _, _ = train_fine(adj_t, split_edge, args.batch_size, args, evaluator, save_path_model_fine,
                              save_path_model, save_path_arch, num_nodes, device)

        results, sub_count_train, sub_count_test = train_model(adj_t, split_edge, args.batch_size, args, evaluator, save_path_model_test,
                              save_path_model, save_path_arch, num_nodes, device)

        t3 = time.time()
        for key, result in results.items():
            loggers[key].add_result(run, result)

        for key, result in results_fine.items():
            loggers_fine[key].add_result(run, result)

        print('##### Testing on {}/{}'.format(run, args.runs))
        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')

        print('##### Finetuning on {}/{}'.format(run, args.runs))
        for key, result in results_fine.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')

        print('***** Running time for arch_selection: {} and retain: {} at {}/{}'.format(t2 - t1, t3 - t2, run,
                                                                                         args.runs))

    print('##### Final subgraph collection for train \n')
    for i in range(args.num_layers):
        print(sub_count_train[i])
    print('##### Final subgraph collection for test \n')
    for i in range(args.num_layers):
        print(sub_count_test[i])

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    print('##### Final Finetune result \n')
    for key in loggers_fine.keys():
        print(key)
        loggers_fine[key].print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--test_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='ogbl-ddi')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--arch_layers', type=int, default=3)
    parser.add_argument('--lin_layers', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--arch_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cat_type', type=str, default='multi', help='multi | concat')
    parser.add_argument('--batch_size', type=int, default=10 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_fine', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    main(data.num_nodes, adj_t, split_edge, args, device)
