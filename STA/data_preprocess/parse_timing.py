import time
import sys
line_array = []
targetPath = sys.argv[1]  
cellName = sys.argv[2] + '/' 
whetherFF = sys.argv[3] 
with open(targetPath, 'r', encoding='utf-8') as f:
    for line in f:
        line_array.append(line.strip())
f.close()
_count = 0
ifStartPoint = 0
for line_index in range(0, len(line_array)):
    if '--------' in line_array[line_index]:
        _count += 1
    if _count == 5:
        break
    if _count != 4:
        continue
            
    split_line = line_array[line_index].split()
    if len(split_line) == 4 and cellName in split_line[3]:
        if '----' in line_array[line_index + 1]:
            whetherFF = -1
            break
        if whetherFF == '1':
                whetherFF = 1
                outputNet = line_array[line_index + 1].split()[-1]
        elif whetherFF == '2':
                if '------' in line_array[line_index - 2]:
                     whetherFF = 2
                     outputNet = line_array[line_index + 1].split()[-1]
                else:
                     ifStartPoint = 1
                     whetherFF = 2
                     outputNet = line_array[line_index + 1].split()[-1]
                     logic_delay = split_line[0]
                     inputNet = line_array[line_index - 2].split()[-1]
        else:
            whetherFF = 0
            outputNet = line_array[line_index + 1].split()[-1]
            logic_delay = split_line[0]
            inputNet = line_array[line_index - 2].split()[-1]
        break
'''
-1 (end point)

1 (FF)
output_Net

2 (BRAM)
output_Net
or
output_Net
input_Net
logic_delay

0 (other)
output_Net
input_Net
logic_delay
'''
with open(targetPath, 'w', encoding='utf-8') as f:
    if whetherFF == -1:
        f.write(str(whetherFF) + '\n')
    elif whetherFF == 1:
        f.write(str(whetherFF) + '\n')
        f.write(outputNet + '\n')
    elif whetherFF == 2:
         if ifStartPoint == 0:
               f.write(str(whetherFF) + '\n')
               f.write(outputNet + '\n')
         else:
               f.write(str(whetherFF) + '\n')
               f.write(outputNet + '\n')
               f.write(inputNet + '\n')
               f.write(logic_delay + '\n')
              
    else:
        f.write(str(whetherFF) + '\n')
        f.write(outputNet + '\n')
        f.write(inputNet + '\n')
        f.write(logic_delay + '\n')
f.close()
