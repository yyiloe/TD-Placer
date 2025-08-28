'''written at 2025/7/5'''
import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

class NetPinDataset(Dataset):
    def __init__(self, file_name, type_logic_delay):
        with open(file_name, 'rb') as f:
            self.dataSource = pickle.load(f)
        f.close()
        self.id2name = [cell for cell in self.dataSource]
        self.flattenData = []
        self.flattenData2NetId = []
        self.flattenData2NetIdIn = []
        self.type_logic_delay = type_logic_delay
        
        for cell in self.dataSource:
            for node in self.dataSource[cell]['outNet']:
                if node[18] == 1:
                    self.flattenData.append(node)
                    

        for cellId in tqdm(range(0, len(self.dataSource))):
            for count in range(0, len(self.dataSource[self.id2name[cellId]]['outNet']) - 1):
                self.flattenData2NetId.append(cellId)
                self.flattenData2NetIdIn.append(count)
        assert(len(self.flattenData2NetIdIn) == len(self.flattenData))
            
    def __getitem__(self, index):
        #netId, net_in_index = self.getNetId(index)
        cellId = self.flattenData2NetId[index]
        net_in_index = self.flattenData2NetIdIn[index]
        oneNetInfo = self.dataSource[self.id2name[cellId]]['outNet']
        net = [oneNetInfo[0][0], oneNetInfo[0][1], oneNetInfo[0][2]]
        nodes = []
        in_nodes = []
        start_node = []
        end_node = []
        in_start_node = []
        in_end_node = []
        
        for item in range(0, len(oneNetInfo)):
            if item == len(oneNetInfo) - 1:
                temp_n = [oneNetInfo[item][4], oneNetInfo[item][5]]
                #temp_n = [0, 0]
                temp_n.extend(oneNetInfo[item][8:19])
                temp_n.append(oneNetInfo[0][3])
            else:
                start_node.append(len(oneNetInfo) - 1)
                end_node.append(item)
                temp_n = oneNetInfo[item][6:19]
                #temp_n = [abs(oneNetInfo[item][6] - oneNetInfo[item][4]), abs(oneNetInfo[item][7] - oneNetInfo[item][5])]
                #temp_n.extend(oneNetInfo[item][8:19])
                temp_n.append(oneNetInfo[0][3])
            nodes.append(temp_n)
        pin2pin = [abs(self.dataSource[self.id2name[cellId]]['outNet'][-1][4] - self.flattenData[index][6]), abs(self.dataSource[self.id2name[cellId]]['outNet'][-1][5]- self.flattenData[index][7])]
        pin2pin.append(net_in_index)
        '''pin2pin.extend(
            self.flattenData[index][6:8] + self.flattenData[index][19:23])'''
        pin2pin.extend(self.flattenData[index][19:23])
        ### process for inputNet
        if 'inNet' in self.dataSource[self.id2name[cellId]]:
            cell_type = ''
            logic_delay = 0
            inNetInfo = self.dataSource[self.id2name[cellId]]['inNet']
            #pin2pin[2] = len(oneNetInfo)
            for item in range(0, len(inNetInfo)):
                if item == len(oneNetInfo) - 1:
                    temp_n = [inNetInfo[item][0], inNetInfo[item][1]]
                    #temp_n = [0, 0]
                    temp_n.extend(inNetInfo[item][4:-1])
                    in_nodes.append(temp_n)
                else:
                    if inNetInfo[item][-1] != 0:
                        first_non_zero = next((i for i in range(4, 14) if inNetInfo[item][i] != 0), None)
                        str_dict = str(first_non_zero)
                        cell_type = str_dict
                        if str_dict != '4':
                            #logic_delay = inNetInfo[item][-1] - self.type_logic_delay[str_dict]
                            logic_delay = inNetInfo[item][-1]
                        else:
                            #logic_delay = inNetInfo[item][-1] - self.type_logic_delay[str_dict][str(inNetInfo[item][first_non_zero])]
                            logic_delay = inNetInfo[item][-1]
                            cell_type = cell_type + '.' + str(inNetInfo[item][first_non_zero])
                        #inNetInfo[item][-1] = 1
                    in_start_node.append(len(inNetInfo) - 1)
                    in_end_node.append(item)
                    #temp_n = [abs(inNetInfo[item][0] - inNetInfo[item][2]), abs(inNetInfo[item][1] - inNetInfo[item][3])]
                    #temp_n.extend(inNetInfo[item][4:-1])
                    in_nodes.append(inNetInfo[item][2:-1])
                    in_nodes.append(temp_n)
                    #in_nodes.append(inNetInfo[item][4:-1])
            ## last return variable reflect whether has inputNet
            return nodes, net, pin2pin, (start_node, end_node), self.flattenData[index][-1], in_nodes, (in_start_node, in_end_node), logic_delay, cell_type
        else:
            return nodes, net, pin2pin, (start_node, end_node), self.flattenData[index][-1], -1, -1, False, -1        
                
        

    def __len__(self):
        return len(self.flattenData)