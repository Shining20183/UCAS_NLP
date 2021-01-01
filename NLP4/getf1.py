from input_data import Dataloader
import sklearn
import numpy as np

test_dataloader = Dataloader('SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
test_text, test_label = test_dataloader.load_data()
test_label = np.array(test_label)

predict = []
with open('model_pre.txt', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.strip('\n')
        predict.append(int(line))
print(predict)
f1 = sklearn.metrics.f1_score(test_label, predict, average='macro')
print(f1)
