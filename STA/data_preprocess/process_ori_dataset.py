## Convert the net_info extracted from TCL into PKL format for storage.
#.pkl:{
#'cell:'{
#inputNet:[[cellx_driver,celly_driver,cellx_load,celly_load,if LUT, if FF, if DSP, if RAMB, if MUX, if IO, if clockbuffer, if CARRY, if shifter, if LUTRAM, if 'IN', Avg.p_Cong, if target cell],...[out_put_pin]]
#outputNet:[[width,length,Fanall,Avg.r Cong,cellx_driver,celly_driver,cellx_load,celly_load,if LUT, if FF, if DSP, if RAMB, if MUX, if IO, if clockbuffer, if CARRY, if shifter, if LUTRAM, if 'IN',if I/O Crossing,if DSP Crossing,if BRAM Crossing,Avg.p_Cong,Delay],...[out_put_pin]]
#}
#}

import pickle
import numpy as np
from tqdm import tqdm

## get VCU095 deviceinfo

def processCong(deviceInfoPath, netInfoCong, binWidth, binHeight, site_loc_dict, piceoff_loc_dict):
    def getCellXY(locDriven):
        return site_loc_dict[locDriven][0], site_loc_dict[locDriven][1]
    

    def getGridXY(cellx, celly, startX, startY, binWidth, binHeight):
        coord_offsetX = cellx - startX
        coord_offsetY = celly - startY
        binIdX = int(coord_offsetX / binWidth)
        binIdY = int(coord_offsetY / binHeight)
        if binIdY < 0:
            binIdY = 0
        if binIdX < 0:
            binIdX = 0
        return binIdX, binIdY
    

    def createBinGridForRcong(deviceInfoPath, binWidth, binHeight):
        min_x = min_y = 100000.0
        max_x = max_y = -100000.0
        with open(deviceInfoPath + 'exportSiteLocation', 'r') as deviceInfo:
            for line in deviceInfo:
                line_s = line.strip().split(' ')
                centry_x = float(line_s[11])
                centry_y = float(line_s[13])
                if min_x > centry_x:
                    min_x = centry_x
                if min_y > centry_y:
                    min_y = centry_y
                if max_x < centry_x:
                    max_x = centry_x
                if max_y < centry_y:
                    max_y = centry_y
        deviceInfo.close()

        startX = round(min_x)
        startY = round(min_y)
        endX = round(max_x)
        endY = round(max_y)

        binGrid = []
        i = 0
        eps = 1e-6
        for curBottomY in np.arange(startY, endY - eps, binHeight):
            row = []
            j = 0
            for curBottomX in np.arange(startX, endX - eps, binWidth):
                item = [0] * 8
                item[0] = i
                item[1] = j
                item[2] = curBottomX
                item[3] = curBottomY
                item[4] = curBottomX + binWidth
                item[5] = curBottomY + binHeight
                row.append(item)
                j += 1
            binGrid.append(row)
        return binGrid, startX, startY, binWidth, binHeight
    
    def getCongGrid(netInfoCong, binGrid, startX, startY, binWidth, binHeight):
        with open(netInfoCong, 'r') as net_info:
            for line in net_info:
                line_s = line.strip().split(' ')
                if line_s[0] == 'curNet=>':
                    net_r = net_u = -10000000.0
                    net_l = net_d = 10000000.0
                if line_s[0] == 'inPin=>' or line_s[0] == 'outPin=>':
                    offset_x = 0
                    offset_y = 0
                    if line_s[5] == 'PCIE_3_1':
                        pin_name = line_s[1].split('/')[-1]
                        if pin_name in piceoff_loc_dict:
                            offset_x = piceoff_loc_dict[pin_name][0]
                            offset_y = piceoff_loc_dict[pin_name][1]
                    cell_x, cell_y = getCellXY(line_s[3])
                    cell_x = cell_x + offset_x
                    cell_y = cell_y + offset_y
                    net_l = min(cell_x, net_l)
                    net_r = max(cell_x, net_r)
                    net_d = min(cell_y, net_d)
                    net_u = max(cell_y, net_u)
                if line_s[0] == 'fanall=>':
                    fanall = int(line_s[1])
                    if fanall <= 1:
                        continue
                    leftBinX, bottomBinY = getGridXY(net_l, net_d, startX, startY, binWidth, binHeight)
                    rightBinX, topBinY = getGridXY(net_r, net_u, startX, startY, binWidth, binHeight)
                    if topBinY >= len(binGrid):
                        topBinY = len(binGrid) - 1
                    if rightBinX >= len(binGrid[0]):
                        rightBinX = len(binGrid[0]) - 1
                    if bottomBinY >= len(binGrid):
                        bottomBinY = len(binGrid) - 1
                    if leftBinX >= len(binGrid[0]):
                        leftBinX = len(binGrid[0]) - 1
                    totW = abs(net_r - net_l) + 0.4 * abs(net_u - net_d) + 0.5
                    if fanall < 10:
                        totW *= 1.06
                    elif fanall < 20:
                        totW *= 1.2
                    elif fanall < 30:
                        totW *= 1.4
                    elif fanall  < 50:
                        totW *= 1.6
                    elif fanall < 100:
                        totW *= 1.8
                    elif fanall < 200:
                        totW *= 2.1
                    else:
                        totW *= 3.0

                    numGCell = (rightBinX - leftBinX + 1) * (topBinY - bottomBinY + 1) * binHeight * binWidth
                    indW = totW / numGCell
                    for i in range(leftBinX, rightBinX + 1):
                        for j in range(bottomBinY, topBinY + 1):
                            binGrid[j][i][7] += indW
                            binGrid[j][i][6] += fanall
                    net_r = net_u = -10000000.0
                    net_l = net_d = 10000000.0
        net_info.close()
        return binGrid
    

    binGrid, startX, startY, binWidth, binHeight = createBinGridForRcong(deviceInfoPath, binWidth, binHeight) 
    binGrid = getCongGrid(netInfoCong, binGrid, startX, startY, binWidth, binHeight)
    
    return binGrid, startX, startY









def processDeviceInfo(path, pciePath):
    site_loc_dict = {}
    piceoff_loc_dict = {}
    with open(path, 'r') as deviceLoc:
        for line in deviceLoc:
            line_s = line.strip().split(' ')
            assert len(line_s) == 16
            if line_s[1] not in site_loc_dict:
                site_loc_dict[line_s[1]] = [float(line_s[11]), float(line_s[13])]
    with open(pciePath, 'r') as piceLoc:
        for line in piceLoc:
            line_s = line.strip().split(' ')
            if line_s[1] not in piceoff_loc_dict:
                piceoff_loc_dict[line_s[1]] = [int(line_s[3]), int(line_s[5])]
    return site_loc_dict, piceoff_loc_dict






def parseNetinfo(ori_path_file, target_path_file, site_loc_dict, piceoff_loc_dict, cong_Bin_Grid, binWidth, binHeight, startX,startY):
    # get cell loc
    def getCellXY(locDriven):
        return site_loc_dict[locDriven][0], site_loc_dict[locDriven][1]
    def checkCellItemLegal(cell_dict):
        if 'outNet' not in cell_dict:
            return False
        if 'inNet' not in cell_dict:
            if cell_dict['outNet'][-1][11] != 1 and  cell_dict['outNet'][-1][9] != 1:
                return False
            if len(cell_dict['outNet']) <= 1:
                return False
        if 'inNet' in cell_dict and 'outNet' in cell_dict:
            if len(cell_dict['inNet']) <= 1 or len(cell_dict['outNet']) <= 1:
                return False
        for key in cell_dict:
            for item in cell_dict[key]:
                if key == 'inNet':
                    if all(x == 0 for x in item[:4]) or item[-1] == -1:
                        return False
                else:
                    if all(x == 0 for x in item[4:8]) or item[-1] == -1:
                        return False
        return True
    
    def getCellTYpeIndex(typename,net_type):
        if typename == 'FDCE' or typename == 'FDRE' or typename == 'FDSE' \
                or typename == 'FDPE':
                if net_type == 1:
                    return 9, False
                else:
                    return 5, False
        if typename == 'LUT1' or typename == 'LUT2' or typename == 'LUT3' \
                or typename == 'LUT4' or typename == 'LUT5' or typename == 'LUT6' or typename == 'LUT6_2':
            if '_' not in typename:
                LUT_type = int(typename[-1])
                return LUT_type, True
            else:
                return 7, True

        if 'DSP' in typename:
            if net_type == 1:
                return 10, False
            else:
                return 6, False
        if 'RAMB' in typename:
            if net_type == 1:
                return 11, False
            else:
                return 7, False
        if 'MUX' in typename:
            if net_type == 1:
                return 12, False
            else:
                return 8, False
        if 'BUFG' in typename:
            if net_type == 1:
                return 14, False
            else:
                return 10, False
        if typename == 'CARRY8':
            if net_type == 1:
                return 15, False
            else:
                return 11, False
        if 'SRL' in typename:
            if net_type == 1:
                return 16, False
            else:
                return 12, False
        if 'RAM' in typename:
            if net_type == 1:
                return 17, False
            else:
                return 13, False
        else:
            if net_type == 1:
                return 13, False
            else:
                return 9, False


    def getNetsWidthLength(nets, netType):
        if netType == 1:
            startIndex = 4
        else:
            startIndex = 0
        minx = nets[-1][startIndex]
        maxx = nets[-1][startIndex]
        miny = nets[-1][startIndex + 1]
        maxy = nets[-1][startIndex + 1]
        
        for index in range(0,len(nets) - 1):
            if nets[index][startIndex + 2] < minx:
                minx = nets[index][startIndex + 2]
            if nets[index][startIndex + 2] > maxx:
                maxx = nets[index][startIndex + 2]
            if nets[index][startIndex + 3] < miny:
                miny = nets[index][startIndex + 3]
            if nets[index][startIndex + 3] > maxy:
                maxy = nets[index][startIndex + 3]
        return abs(maxy - miny), abs(maxx - minx), minx, miny, maxx, maxy
    

    def getWhetherCross(cellx_load,cellx_driver):
        minx = min(cellx_load, cellx_driver)
        maxx = max(cellx_load, cellx_driver)
        ##judge IO
        whetherIO = 0
        IOX4 = [7.5, 33.5, 51.5, 76.5]

        for i in range(0, 4):
            if minx < IOX4[i] < maxx:
                whetherIO = 1
                break
        ##judge DSP
        whetherDSP = 0
        DSPX4 = [15.5, 69.5]
        for i in range(0, 2):
            if minx < DSPX4[i] < maxx:
                whetherDSP = 1
                break

        whetherRAMB = 0
        RAMBX18 = [5.75, 10.75, 13.75, 17.75, 22.75, 26.75, 31.75, 36.75, 41.75, 45.75, 50.75, 55.75, 60.75, 64.75,
               74.75, 80.75]
        for i in range(0, 16):
            if minx < RAMBX18[i] < maxx:
                whetherRAMB = 1
                break
        return whetherIO , whetherDSP , whetherRAMB 
    
    def processCongForAllNets(cong_Bin_Grid, target_dataset, binWidth, binHeight, startX, startY):
        def getGridXY(cellx, celly, startX, startY, binWidth, binHeight, cong_Bin_Grid):
            coord_offsetX = cellx - startX
            coord_offsetY = celly - startY
            binIdX = int(coord_offsetX / binWidth)
            binIdY = int(coord_offsetY / binHeight)
            if binIdY < 0:
                binIdY = 0
            if binIdX < 0:
                binIdX = 0
            if binIdY >= len(cong_Bin_Grid):
                binIdY = len(cong_Bin_Grid) - 1
            if binIdX >= len(cong_Bin_Grid[0]):
                binIdX = len(cong_Bin_Grid[0]) - 1
            return binIdX, binIdY
        
        
        for cell in target_dataset:
            for net in target_dataset[cell]:
                if net == 'outNet':
                    _,_, minx, miny, maxx, maxy = getNetsWidthLength(target_dataset[cell][net], 1)
                    leftBinX, bottomBinY = getGridXY(minx, miny, startX, startY, binWidth, binHeight, cong_Bin_Grid)
                    rightBinX, topBinY = getGridXY(maxx, maxy, startX, startY, binWidth, binHeight, cong_Bin_Grid)
                    total_r = 0
                    count = 0

                    for i in range(leftBinX, rightBinX + 1):
                        for j in range(bottomBinY, topBinY + 1):
                            total_r += cong_Bin_Grid[j][i][7]
                            count += 1
                    aveRCong = total_r / count
                    

                    cell_x_driver, cell_y_driver = target_dataset[cell][net][-1][4], target_dataset[cell][net][-1][5]
                    d_x, d_y = getGridXY(cell_x_driver, cell_y_driver, startX, startY, binWidth, binHeight, cong_Bin_Grid)
                    for pin_item in target_dataset[cell][net]:
                        pin_item[3] = aveRCong
                        if pin_item[18] != 1:
                            continue
                        n_x, n_y = getGridXY(pin_item[6], pin_item[7], startX, startY, binWidth, binHeight, cong_Bin_Grid)
                        min_x = min(d_x, n_x)
                        max_x = max(d_x, n_x)
                        min_y = min(n_y, d_y)
                        max_y = max(n_y, d_y)
                        total_p = 0
                        count = 0
                        for i in range(min_x, max_x + 1):
                            for j in range(min_y, max_y + 1):
                                total_p += cong_Bin_Grid[j][i][6]
                                count += 1
                        pCong = total_p / count
                        pin_item[22] = pCong / 1e6
                else:
                    _,_, minx, miny, maxx, maxy = getNetsWidthLength(target_dataset[cell][net], 2)
                    leftBinX, bottomBinY = getGridXY(minx, miny, startX, startY, binWidth, binHeight, cong_Bin_Grid)
                    rightBinX, topBinY = getGridXY(maxx, maxy, startX, startY, binWidth, binHeight, cong_Bin_Grid)
                    total_r = 0
                    count = 0

                    for i in range(leftBinX, rightBinX + 1):
                        for j in range(bottomBinY, topBinY + 1):
                            total_r += cong_Bin_Grid[j][i][7]
                            count += 1
                    aveRCong = total_r / count
                    for pin_item in target_dataset[cell][net]:
                        pin_item[15] = aveRCong
                 
        return target_dataset
    


    target_dataset = {}
    #line-by-line processing
    with open(ori_path_file, 'r') as net_info:
        cur_cell = ''
        for line in tqdm(net_info, desc="Processing lines"):
            line_s = line.strip().split(' ')
            if line_s[0] == 'curCell=>':
                #### before get new check legal
                if len(target_dataset) >= 1:
                    last_key = list(target_dataset.keys())[-1]
                    if not checkCellItemLegal(target_dataset[last_key]):
                        del target_dataset[cur_cell]
                        #pass
                ####
                net_type = 0 ## 1 outNet; 2 inNet
                target_dataset[line_s[1]] = {}
                cur_cell = line_s[1]
                logic_count = -1
            

            elif line_s[0] == 'outNet=>':
                net_type = 1
                target_dataset[cur_cell]['outNet'] = []
            elif line_s[0] == 'inNet=>':
                net_type = 2
                target_dataset[cur_cell]['inNet'] = []
            

            elif line_s[0] == 'inPin=>' or line_s[0] == 'outPin=>':
                if line_s[3] == '':
                    continue
                if net_type == 1:
                    new_item = [0] * 24
                    cell_x, cell_y = getCellXY(line_s[3])
                    offset_x = 0
                    offset_y = 0
                    if line_s[5] == 'PCIE_3_1':
                        pin_name = line_s[1].split('/')[-1]
                        if pin_name in piceoff_loc_dict:
                            offset_x = piceoff_loc_dict[pin_name][0]
                            offset_y = piceoff_loc_dict[pin_name][1]
                    type_index, ifLUT = getCellTYpeIndex(line_s[5],net_type)
                    if ifLUT:
                        new_item[8] = type_index
                    else:
                        new_item[type_index] = 1
                    
                    if line_s[0] == 'inPin=>':
                        new_item[18] = 1
                        if line_s[-1] == '':
                            new_item[-1] = -1
                        else:
                            new_item[-1] = int(line_s[-1])   ##float or int
                        new_item[6] = offset_x + cell_x
                        new_item[7] = offset_y + cell_y
                    else:
                        new_item[4] = offset_x + cell_x
                        new_item[5] = offset_y + cell_y
                    target_dataset[cur_cell]['outNet'].append(new_item)
                        





                if net_type == 2:
                    new_item = [0] * 17
                    cell_x, cell_y = getCellXY(line_s[3])
                    offset_x = 0
                    offset_y = 0
                    
                    if line_s[5] == 'PCIE_3_1':
                        pin_name = line_s[1].split('/')[-1]
                        if pin_name in piceoff_loc_dict:
                            offset_x = piceoff_loc_dict[pin_name][0]
                            offset_y = piceoff_loc_dict[pin_name][1]
                    
                    type_index, ifLUT = getCellTYpeIndex(line_s[5],net_type)
                    if ifLUT:
                        new_item[4] = type_index
                    else:
                        new_item[type_index] = 1
                        
                    if cur_cell in line_s[1]:
                        logic_count = len(target_dataset[cur_cell]['inNet'])
 #                       if ifLUT and line_s[0] == 'inPin=>':
                            #print(line_s[1])
 #                           new_item[16] = int(line_s[1].split('/')[-1][-1]) + 1
                            #print(new_item[16])
                    if line_s[0] == 'inPin=>':
                        new_item[14] = 1
                        new_item[2] = offset_x + cell_x
                        new_item[3] = offset_y + cell_y
                    else:
                        new_item[0] = offset_x + cell_x
                        new_item[1] = offset_y + cell_y
                    target_dataset[cur_cell]['inNet'].append(new_item)
            elif line_s[0] == 'fanall=>':
                if net_type == 1:
                    #### width,length,Fanout,if I/O Crossing,if DSP Crossing,if BRAM Crossing
                    width, length, _ ,_ ,_ ,_ = getNetsWidthLength(target_dataset[cur_cell]['outNet'], net_type)
                    for item in target_dataset[cur_cell]['outNet']:
                        item[0] = width
                        item[1] = length
                        item[2] = int(line_s[1])
                        if item[18] != 1:
                            continue
                        item[19], item[20], item[21] = getWhetherCross(item[6], target_dataset[cur_cell]['outNet'][-1][4])
                    
            
            elif line_s[0] == 'logicDelay=>':
                if logic_count != -1:
                    if line_s[1] == '':
                        target_dataset[cur_cell]['inNet'][logic_count][-1] = -1
                    else:
                        target_dataset[cur_cell]['inNet'][logic_count][-1] = int(float(line_s[1]))
            
            else:
                continue
    net_info.close()
    target_dataset = processCongForAllNets(cong_Bin_Grid, target_dataset,  binWidth, binHeight ,startX, startY)

    with open(target_path_file, 'wb') as dataset:
        pickle.dump(target_dataset, dataset)
    dataset.close()
    

    return 0
                    
                
                    
    
    


if __name__ == '__main__':
    target_path_file = '../dataset/processed_data/net_info.pkl'
    ori_path_file = '../dataset/ori_data/net_info'
    device_info_path = './device_info/'
    net_info_cong = '../dataset/ori_data/net_info_cong'
    binWidth = 2.0
    binHeight = 2.0

    site_loc_dict = {}
    piceoff_loc_dict = {}

    site_loc_dict, piceoff_loc_dict = processDeviceInfo(device_info_path + 'exportSiteLocation', device_info_path + 'PCIEPin2SwXY')
    binGrid, startx, starty = processCong(device_info_path, net_info_cong, binWidth, binHeight, site_loc_dict, piceoff_loc_dict)
    parseNetinfo(ori_path_file, target_path_file, site_loc_dict, piceoff_loc_dict, binGrid, binWidth, binHeight, startx, starty)
    