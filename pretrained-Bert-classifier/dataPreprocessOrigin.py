#encoding=utf-8

inputpath="TestA.txt"
# trainpath='train.tsv'
trainpath='test.tsv'
# devpath='dev.tsv'
trainfile=open(trainpath,'w',encoding='utf-8')
# devfile=open(devpath,'w',encoding='utf-8')
flag=0
with open(inputpath,'r') as f:
    trainfile.write("index\tquestion\tsentence\tlabel\n")
    # devfile.write("index\tquestion\tsentence\tlabel\n")

    # idx1=0
    # idx2=0
    idx=0
    for line in f:
        if flag==0:
            flag=1
            continue
        item = line.strip().split('\t')
        # idx = item[0]
        label = 0
        sen = item[1]
        # if idx1<3450:
        #     trainfile.write(str(idx1)+'\t'+str(sen)+'\t'+' '+'\t'+str(int(label))+'\n')
        trainfile.write(str(idx)+'\t'+str(sen)+'\t'+' '+'\t'+str(int(label))+'\n')
        idx+=1
            # idx1+=1
        # else:
        #     devfile.write(str(idx2) + '\t' + str(sen) + '\t' + ' ' + '\t' + str(int(label)) + '\n')
        #     idx2 += 1




