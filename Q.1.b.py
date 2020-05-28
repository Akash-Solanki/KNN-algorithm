import numpy as np
import pandas as pd



df1 = pd.read_csv('mnist_train.csv')
df2 = pd.read_csv('mnist_test.csv')
arr1=df1.values
arr2=df2.values
data=arr1[:,1:]; #training data set
cla=arr1[:,0]; #class label for training dataset
test=arr2[:,1:]; #test data set



g=100; #for g samples in test set
hist_test=[]; #finding the histogram of each image in test set
for i in range(0,len(test)):
    a,b=np.histogram(test[i],bins=np.arange(0,256))
    hist_test.append(a)
hist_train=[]; #finding the histogram of each image in training set
for i in range(0,len(data)):
    e,f=np.histogram(data[i],bins=np.arange(0,256))
    hist_train.append(e)

differ=[] #finding the difference between histogram of each test sample with all other training set
for j in range(0,g):
    a=[]
    for i in range(0,len(hist_train)):
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
t= np.argsort(finalt,axis=0); #sorting the array to find the min k distances
t[0:k]

uhu=[]
for i in range(0,K):
    f=[];
    for k in t[i]:
        f.append(cla[k]);
    uhu.append(f);
uhu

som=np.array(uhu)

fin_arr=[]; #classifying which test sample belong to which class 
for i in range(0,len(som.T)):
    counts=np.argmax(np.bincount(som[:,i]));
    fin_arr.append(counts)
   
print("predicted digits are"+" "+ str(fin_arr))

acc=(np.sum(np.array(fin_arr)==np.array(arr2[:g,0])))/g #accuracy of he model
print('the accuracy of the model is' +' '+ str(acc*100)+'%')




    

