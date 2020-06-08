import glob
import re
import numpy as np

for result_file in glob.glob("results/*.txt"):
#    print(result_file)
    r = re.match("results\/(\d+)\.(\d+)\.(\d+)\.(\d+)\.txt", result_file)    
    ecc = int(r.group(1))
    mode = int(r.group(2))
    noise = int(r.group(3))
    gap = int(r.group(4))
    with open(result_file, "r") as F:  
#        print(result_file)
        row = [ecc, mode, noise, gap]        
        row_values = []        
        for line in F.readlines():
            if re.match("(\d+\.\d+)\t(\d+\.\d+)\t(\d+\.\d+)\t(\d+\.\d+)", line):                                
                row_values.append(map(float, line.strip().split("\t")))
                
        row_values = np.array(row_values)

        mean_train_acc = np.mean(row_values[:,0])
        mean_test_acc = np.mean(row_values[:,2])
        std_train_acc = np.std(row_values[:,0])
        std_test_acc = np.std(row_values[:,2])            
        row += [mean_train_acc, mean_test_acc, std_train_acc, std_test_acc]
	row += [row_values[:, 0].shape[0]]
        print("\t".join(map(str, row)))

