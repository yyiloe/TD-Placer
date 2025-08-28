import os
import torch

class Config:
    def __init__(self):
        self.pretrained = True
        self.autoWeightForLoss = True
        self.whetherTransfer = False
        self.whetherMTL = False
        self.data_root = "your path/TD-Placer/STA/dataset/processed_data"
        self.data_root_transfer = "your path/TD-Placer/STA/dataset/processed_data/transfer"
        if self.whetherTransfer == False:
            self.train_dir = os.path.join(self.data_root, "train")
            self.val_dir = os.path.join(self.data_root, "val")
            self.test_dir = os.path.join(self.data_root, "test")
        else:
            self.train_dir = os.path.join(self.data_root_transfer, "train")
            self.val_dir = os.path.join(self.data_root_transfer, "val")
            self.test_dir = os.path.join(self.data_root_transfer, "test")
        self.save_dir = "your path/TD-Placer/STA/model/weight"


        self.batch_size = 2048
        self.num_epochs = 600

        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.lr_scheduler = 'cosine'  # 可选：'step', 'cosine', 'none'
        self.step_size = 50
        self.gamma = 0.95

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"
        self.seed = 42
        
        #self.save_name = f"{self.model_name}_best.pth"
        
        self.model_param = {"in_feats": 14, "hidden_feats": 24, "net_feats": 4, "out_feats": 1, "whetherCrossAttention": True, "whetherGAT": True, "depth": 1}
        self.type_logic_delay = {'4': {'1':98, '2':130, '3':128, '4':122, '5':117, '6':118}, '6': 595, '8': 69, '11': 148 }
        self.figure_save_dir = 'your path/TD-Placer/STA/figure'

    def display(self):
        print("========= Configurations =========")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        print("==================================")
