#encoding=utf-8
import json
testpath='train.tsv'

testfile=open(testpath,'w',encoding='utf-8')
with open("TrainA.txt", 'r') as file:

    # dataList=data['data'][:3450]
    testfile.write("index\tquestion\tsentence\tlabel\n")
    idx=0
    flag=0
    for line in file:
        if flag==0:
            flag=1
            continue
        line=line.strip().split('\t')
        label=int(line[1])
        wordList=line[2]
        if label==0 or label==1:
            testfile.write(str(idx)+'\t'+str(wordList)+'\t'+' '+'\t'+str(int(label))+'\n')
            idx+=1
        elif label ==2:
            for i in range(8):
                testfile.write(str(idx) + '\t' + str(wordList) + '\t' + ' ' + '\t' + str(int(label)) + '\n')
                idx+=1
        elif label ==3:
            for i in range(9):
                testfile.write(str(idx) + '\t' + str(' '.join(wordList)) + '\t' + ' ' + '\t' + str(int(label)) + '\n')
                idx+=1