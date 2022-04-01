import random
from types import new_class
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

bases={'A':np.array([0,0,0,1]), 'C':np.array([0,0,1,0]), 'G':np.array([0,1,0,0]), 'T':np.array([1,0,0,0])}

def Kmers_funct(seq, size):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

def one_hot_encode_2(y,num_classes):
    y_encoded=[]
    for value in y:
        letter = [0 for _ in range(num_classes)]
        letter[value] = 1
        y_encoded.append(letter)
    return np.array(y_encoded,dtype=np.float16)

def release_list(a):
   del a[:]
   del a


def selectref(el,pa,pc,pg,pt):

    dicref={'U':['T'],
    'R':['A','G'],
    'Y':['C','T'],
    'S':['G','C'],
    'W':['A','T'],
    'K':['G','T'],
    'M':['A','C'],
    'B':['C','G','T'],
    'D':['A','G','T'],
    'H':['A','C','T'],
    'V':['A','C','G'],
    'N':['A','T','G','C'],
    }

    dicprobs={'U':[1],
    'R':[pa/(pa+pg),pg/(pa+pg)],
    'Y':[pc/(pc+pt),pt/(pc+pt)],
    'S':[pg/(pc+pg),pc/(pc+pg)],
    'W':[pa/(pa+pt),pt/(pa+pt)],
    'K':[pg/(pt+pg),pt/(pt+pg)],
    'M':[pa/(pa+pc),pc/(pa+pc)],
    'B':[pc/(pc+pg+pt),pg/(pc+pg+pt),pt/(pc+pg+pt)],
    'D':[pa/(pa+pg+pt),pg/(pa+pg+pt),pt/(pa+pg+pt)],
    'H':[pa/(pc+pa+pt),pc/(pc+pa+pt),pt/(pc+pa+pt)],
    'V':[pa/(pc+pg+pa),pc/(pc+pg+pa),pg/(pc+pg+pa)],
    'N':[pa/(pc+pg+pa+pt),pt/(pc+pg+pa+pt),pg/(pc+pg+pa+pt),pc/(pc+pg+pa+pt)]}

    # print(dicprobs[el])

    return np.random.choice(dicref[el],p=dicprobs[el])
  # return np.random.choice(dicref[el],p=dicprobs[el])


def generate_dataset():
    
    cv = CountVectorizer()
    unique_elems=dict()

    all_sequences=[]


    for sequence in SeqIO.parse('ncbi_16s_18s_merged.fasta', "fasta"):
        unique_elems[sequence.description.split()[1]]=0
        seq=""

        base_count=dict()
        base_count['A']=0
        base_count['C']=0
        base_count['G']=0
        base_count['T']=0
        base_count['others']=0

        for el in sequence.seq:
            if el not in base_count.keys():

                base_count['others']+=1
            else:
                base_count[el]+=1
    
        na=base_count['A']
        nc=base_count['C']
        ng=base_count['G']
        nt=base_count['T']

        pa=float(na/(na+nc+ng+nt))
        pc=float(nc/(na+nc+ng+nt))
        pg=float(ng/(na+nc+ng+nt))
        pt=float(nt/(na+nc+ng+nt))

        for el in sequence.seq:
            if el not in base_count.keys():

                seq += (selectref(el,pa,pc,pg,pt))
            else:
                seq+=el
    
        words = Kmers_funct(seq, size=6)
        joined_sentence = ' '.join(words)
        all_sequences.append(joined_sentence)

    X=cv.fit_transform(all_sequences).toarray()

    i=0
    x_data=[]
    y_data=[]
    for el in unique_elems.keys():
        unique_elems[el]=i
        i+=1

    i=0
    for sequence in SeqIO.parse('ncbi_16s_18s_merged.fasta', "fasta"):
        x_data.append(X[i])
        y_data.append(unique_elems[sequence.description.split()[1]])
        i+=1

    lm=len(unique_elems)

    class_count=dict()

    for i in range(lm):
        class_count[i]=0

    for el in y_data:
        class_count[el]+=1

    dominant_classes=([idx for idx, element in enumerate(class_count.values()) if element>50])

    print(len(dominant_classes))

    new_class_map=dict()

    for idx,element in enumerate(dominant_classes):
        new_class_map[element]=idx

  
    x_data_new=[]
    y_data_new=[]

    for i in range(len(y_data)):
        if y_data[i] in dominant_classes:
            x_data_new.append(x_data[i])
            y_data_new.append(new_class_map[y_data[i]])



    release_list(x_data)
    release_list(y_data)

    x_data,y_data=shuffle(x_data_new,y_data_new)

    x_train=x_data[:int(len(x_data)*0.8)]
    y_train=y_data[:int(len(y_data)*0.8)]


    x_test=x_data[int(len(x_data)*0.8):]
    y_test=y_data[int(len(y_data)*0.8):]

    print(len(x_train))

    # y_train=one_hot_encode_2(y_train,len(dominant_classes))
    # y_test=one_hot_encode_2(y_test,len(dominant_classes))

    x_train=np.array(x_train,dtype=np.float32)
    x_train=np.reshape(x_train,(-1,1,4096))

    x_test=np.array(x_test,dtype=np.float32)
    x_test=np.reshape(x_test,(-1,1,4096))

    # return x_train,y_train,x_test,y_test

    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('x_test',x_test)
    np.save('y_test',y_test)

def load_dataset():
    x_train=np.load('x_train.npy')
    y_train=np.load('y_train.npy')
    x_test=np.load('x_test.npy')
    y_test=np.load('y_test.npy')

    return x_train,y_train,x_test,y_test
# x1,y1,x2,y2=generate_dataset()

# print(x1.shape)
# print(y1.shape)
# print(x2.shape)
# print(y2.shape)

# generate_dataset()

