from model.model import GATEncoder, NetLogicMLP
import torch
from dataset.dataset import NetPinDataset
from torch_geometric.data import Data, Batch
from config.config import Config
import random
import numpy as np
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
    
if __name__ == "__main__":
    cfg = Config()
    cfg.display()
    set_seed(cfg.seed)
    test_dataset = NetPinDataset(cfg.test_dir + '/test.pkl', cfg.type_logic_delay)
    model_GAT = GATEncoder(in_feats=cfg.model_param['in_feats'], hidden_feats=cfg.model_param['hidden_feats'], num_layers=cfg.model_param['depth'])
    model_GAT.load_state_dict(torch.load("./model/weight/gcn.pkl"))
    model_NetLogicPredict = NetLogicMLP(hidden_feats=cfg.model_param['hidden_feats'], net_feats=cfg.model_param['net_feats'], out_feats=cfg.model_param['out_feats'], depth=cfg.model_param['depth'])
    model_NetLogicPredict.load_state_dict(torch.load("./model/weight/netLogicMLP.pkl"))
    
    model_GAT.to(cfg.device)
    model_NetLogicPredict.to(cfg.device)
    
    model_GAT.eval()
    model_NetLogicPredict.eval()
    
    batch_data, batch_net, batch_pin2pin, batch_netDelay, batch_data_in, batch_logic_delay, batch_whether_calculate_logic_delay, batch_cell_type = getBatchData(test_dataset,0, cfg.batch_size )
    batch_data = batch_data.to(cfg.device)
    batch_net = batch_net.to(cfg.device)
    batch_pin2pin = batch_pin2pin.to(cfg.device)
    batch_netDelay = batch_netDelay.to(cfg.device)
    batch_data_in = batch_data_in.to(cfg.device)
    batch_logic_delay = batch_logic_delay.to(cfg.device)
    batch_whether_calculate_logic_delay = batch_whether_calculate_logic_delay.to(cfg.device)
    
    batchNetInfo_out = model_GAT(batch_data.x, batch_data.edge_index, batch_data.batch)
    batchNetInfo_in = model_GAT(batch_data_in.x, batch_data_in.edge_index, batch_data_in.batch)

    traced_model_GAT = torch.jit.trace(model_GAT, (batch_data.x, batch_data.edge_index, batch_data.batch))
    traced_model_netLogicDelay = torch.jit.trace(model_NetLogicPredict, (batchNetInfo_out, batchNetInfo_in, batch_net, batch_pin2pin))
    traced_model_GAT.save('netInfoExtract.pt')
    traced_model_netLogicDelay.save('netLogicDelayRegression.pt')
    
    
