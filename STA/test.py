import os

import torch
from torch import nn
from model.model import NetLogicDelayPredict
from model.AutomaticWeightedLoss import AutomaticWeightedLoss
from dataset.dataset import NetPinDataset
from torch_geometric.data import Data, Batch
from config.config import Config
from tqdm import tqdm
import random
import numpy as np

import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def getBatchData(dataset, id, batch):
    batch_data = []
    batch_net = []
    batch_pin2pin = []
    batch_netDelay = []
    batch_data_in = []
    batch_logic_delay = []
    batch_whether_calculate_logic_delay = []
    batch_cell_type = []
    for i in range(id, id + batch):
        nodes, net, pin2pin, dirInfo, net_delay, in_nodes, in_dirInfo, logic_delay, cell_type= dataset.__getitem__(i)
        data_temp = Data(torch.tensor(nodes, dtype=torch.float), torch.tensor(dirInfo, dtype=torch.int64))
        batch_data.append(data_temp)
        batch_net.append(net)
        batch_pin2pin.append(pin2pin)
        batch_netDelay.append(net_delay)
        if logic_delay != False:
            data_temp_in = Data(torch.tensor(in_nodes, dtype=torch.float), torch.tensor(in_dirInfo, dtype=torch.int64))
            batch_whether_calculate_logic_delay.append(1)
        else:
            data_temp_in = Data(torch.zeros(2, cfg.model_param['in_feats'], dtype=torch.float), torch.tensor([[0], [1]], dtype=torch.int64))
            batch_whether_calculate_logic_delay.append(0)
        batch_data_in.append(data_temp_in)
        batch_logic_delay.append(logic_delay)
        batch_cell_type.append(cell_type)
        
    return Batch.from_data_list(batch_data), torch.tensor(batch_net, dtype=torch.float), torch.tensor(batch_pin2pin, dtype=torch.float), torch.tensor(batch_netDelay, dtype=torch.float), Batch.from_data_list(batch_data_in), torch.tensor(batch_logic_delay, dtype=torch.float), torch.tensor(batch_whether_calculate_logic_delay, dtype=torch.int64), batch_cell_type


def multitask_loss(logits, net_delays, logic_delays, batch_whether_calculate_logic_delay, mtl_loss_module):
	
    def remove_consecutive_duplicates(values, delays):
        if len(values) == 0:
            return values, delays

        result_values = [values[0]]
        result_delays = [delays[0]]
        for i in range(1, len(values)):
            if values[i].item() != values[i - 1].item():
                result_values.append(values[i])
                result_delays.append(delays[i])
        return torch.stack(result_values), torch.stack(result_delays)

    loss_net = F.mse_loss(logits[:, 0], net_delays)
    
    logic_target = logits[:, 1]
    masked_logic_target = logic_target[batch_whether_calculate_logic_delay == 1]
    masked_logic_delays = logic_delays[batch_whether_calculate_logic_delay == 1]
    #masked_logic_target, masked_logic_delays = remove_consecutive_duplicates(masked_logic_target, masked_logic_delays)
    assert len(masked_logic_target) == len(masked_logic_delays)

    
    loss_logic = F.mse_loss(masked_logic_target, masked_logic_delays)
    
    #####calculate MTL loss: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics #####
    if cfg.autoWeightForLoss:
        loss_weight_sum = mtl_loss_module(loss_net, loss_logic)
        #loss_weight_sum = loss_logic
    else:
        loss_weight_sum = loss_net + loss_logic
    return loss_weight_sum, loss_net, loss_logic, logits[:, 0], masked_logic_target, masked_logic_delays


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True42
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)
    
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8

def getRegressionMetric(all_logits_net_delays, all_logits_logic_delays, all_net_delays, all_logic_delays, all_cell_type):
    mse = mean_squared_error(all_net_delays, all_logits_net_delays)
    mae = mean_absolute_error(all_net_delays, all_logits_net_delays)
    r2 = r2_score(all_net_delays, all_logits_net_delays)

    tqdm.write(f"MSE_net_delay: {mse:.6f}")
    tqdm.write(f"MAE_net_delay: {mae:.6f}")
    tqdm.write(f"R2_net_delay Score: {r2:.6f}")
    
    mse_logic = mean_squared_error(all_logic_delays, all_logits_logic_delays)
    mae_logic = mean_absolute_error(all_logic_delays, all_logits_logic_delays)
    r2_logic = r2_score(all_logic_delays, all_logits_logic_delays)
    mean_value = np.mean(all_logic_delays)
    mean_list = [mean_value] * len(all_logic_delays)
    mean_mae_logic = mean_absolute_error(mean_list, all_logic_delays)
    mean_mse_logic = mean_squared_error(mean_list, all_logic_delays)
    if cfg.whetherTransfer == False and cfg.whetherMTL:
        tqdm.write(f"mean_MAE_logic_delay: {mean_mae_logic}")
        tqdm.write(f"mean_MSE_logic_delay: {mean_mse_logic}")
        tqdm.write(f"MSE_logic_delay: {mse_logic:.6f}")
        tqdm.write(f"MAE_logic_delay: {mae_logic:.6f}")
        tqdm.write(f"R2_logic_delay Score: {r2_logic:.6f}")
    
    #tqdm.write(f"val_mean_logic_delay: {sum([x + 118 for x in all_logic_delays ])/len(all_logic_delays):.6f}")
    '''y = np.array(all_logic_delays)
    tqdm.write(f"标签最大值:{y.max()}")
    tqdm.write(f"标签最小值:{y.min()}")
    tqdm.write(f"标签标准差:{y.std()}")
    tqdm.write(f"标签均值:{y.mean()}")'''

    return r2, r2_logic, mse_logic, mae_logic, mean_mae_logic, mean_mse_logic


if __name__ == "__main__":
    cfg = Config()
    cfg.display()
    set_seed(cfg.seed)
    test_dataset = NetPinDataset(cfg.test_dir + '/test.pkl', cfg.type_logic_delay)
    aw2 = AutomaticWeightedLoss(2)
    aw2 = aw2.to(cfg.device)
    model = NetLogicDelayPredict(in_feats=cfg.model_param['in_feats'], hidden_feats=cfg.model_param['hidden_feats'], net_feats=cfg.model_param['net_feats'], out_feats=cfg.model_param['out_feats'], depth=cfg.model_param['depth'])
    
    model.gcn.load_state_dict(torch.load("./model/weight/gcn.pkl"))
    model.netLogicMLP.load_state_dict(torch.load("./model/weight/netLogicMLP.pkl"))
    model = model.to(cfg.device)
    
    model.eval()
    index = 0
    all_logits_net_delays = []
    all_logits_logic_delays = []
    all_net_delays = []
    all_logic_delays = []
    all_cell_type = []
    with torch.no_grad():
        while index + cfg.batch_size < test_dataset.__len__():
            batch_data, batch_net, batch_pin2pin, batch_netDelay, batch_data_in, batch_logic_delay, batch_whether_calculate_logic_delay, batch_cell_type = getBatchData(test_dataset, index, cfg.batch_size )
            batch_data = batch_data.to(cfg.device)
            batch_net = batch_net.to(cfg.device)
            batch_pin2pin = batch_pin2pin.to(cfg.device)
            batch_netDelay = batch_netDelay.to(cfg.device)
            batch_data_in = batch_data_in.to(cfg.device)
            batch_logic_delay = batch_logic_delay.to(cfg.device)
            batch_whether_calculate_logic_delay = batch_whether_calculate_logic_delay.to(cfg.device)
            logits = model(batch_data.x, batch_data.edge_index, batch_data.batch, batch_net, batch_data_in.x, batch_data_in.batch, batch_data_in.edge_index, batch_pin2pin)
            _, _, _, netDelay_target, masked_logic_target, masked_logic_delays = multitask_loss(logits, batch_netDelay, batch_logic_delay, batch_whether_calculate_logic_delay, aw2)
                
            all_logits_logic_delays.extend(masked_logic_target.detach().cpu().tolist())
            all_logic_delays.extend(masked_logic_delays.detach().cpu().tolist())
            all_net_delays.extend(batch_netDelay.detach().cpu().tolist())
            all_logits_net_delays.extend(netDelay_target.detach().cpu().tolist())
            all_cell_type.extend(batch_cell_type)
            index = index + cfg.batch_size
                ##calculate mae mse R2 for logic and net delay##
        tqdm.write("===============test=================")
        _, _, _, mae_logic_test, mean_mae_logic_test, _ = getRegressionMetric(all_logits_net_delays, all_logits_logic_delays, all_net_delays, all_logic_delays, all_cell_type)
        tqdm.write("=============test end=================")

        
