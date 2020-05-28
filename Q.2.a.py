import numpy as np
import pandas as pd


f=open("differences.txt",'w')
df1 = pd.read_csv('CIFAR10_Train_Data.csv')
df2 = pd.read_csv('CIFAR10_Test_Data.csv')
df3 = pd.read_csv('CIFAR10_Train_Labels.csv')
df4 = pd.read_csv('CIFAR10_Test_Labels.csv')
arr1=df1.values
arr2=df2.values
arr3=df3.values
arr4=df4.values

hist_test=[]; #finding the histogram of each image in test set
for i in range(0,len(arr2)):
    a,b=np.histogram(arr2[i],bins=np.arange(0,256))
    hist_test.append(a)
hist_train=[]; #finding the histogram of each image in training set
for i in range(0,len(arr1)):
    e,f=np.histogram(arr1[i],bins=np.arange(0,256))
    hist_train.append(e)


g=150;   #for g samples in test set
differ=[]  #finding the difference between pixel value of each test sample with all other training set
for j in range(0,g):
    a=[]
    for i in range(0,len(arr1)):
        a.append(hist_train[i]-hist_test[j])
    differ.append(a)
    
final=[] #finding the euclidean distnace between feature vectors
for i in range(0,len(differ)):
    b=[]
    for j in range(0,len(differ[i])):
        su=0
        for k in differ[i][j]:
            su+=k*k
        su=pow(su,0.5)
        
        b.append(su)

    final.append(b)

finall=np.array(final);
finalt=finall.T


K=5; #K is the no. of closest neighbours to choose
t= np.argsort(finalt,axis=0);  #sorting the array to find the min k distances
t[0:k]

fla=arr3.flatten()
uhu=[]
for i in range(0,len(t[0:K])):
    f=[];
    for k in t[i]:
        f.append(fla[k]);
    uhu.append(f);
uhu

som=np.array(uhu)

fin_arr=[]; #classifying which test sample belong to which class 
for i in range(0,len(som.T)):
    counts=np.argmax(np.bincount(som[:,i]));
    fin_arr.append(counts)
   
print("predicted digits are"+" "+ str(fin_arr))

acc=(np.sum(np.array(fin_arr)==np.array(arr4[:g,0])))/g  #accuracy of he model
print('the accuracy of the model for k=5 is' +' '+ str(acc*100))




    

