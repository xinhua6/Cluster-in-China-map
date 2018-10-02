from numpy import *

#读取文件
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


#计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#k-均值聚类算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k): #寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print(centroids)
        for cent in range(k):#更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #沿矩阵的列方向进行均值计算
    return centroids, clusterAssment

#二分k-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error 创建一个初始簇
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            #尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #跟新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json

#计算球面距离
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy


#读取中国省市
import numpy as np
def readFlies(path):
  w = open(path,'r',encoding='UTF-8') #注意编码格式
  lines = w.readlines()
  col=[]
  for k in lines:
    k = k.strip('\n')  #去掉读取中的换行字符
    col.append(k)
  while '' in col:
    col.remove('')    #去掉读取的空格
  return col

#利用百度API获取中国省市的经纬度
from urllib import parse
from urllib.request import urlopen
import hashlib
import json
def get_urt(addtress):
    queryStr = '/geocoder/v2/?address=%s&output=json&ak=MRe5GRd5q51anxVOTr2HpGqqHsSnRDQ3' % addtress
 # 对queryStr进行转码，safe内的保留字符不转换
    encodedStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
 # 在最后直接追加上yoursk
    rawStr = encodedStr + '11991650'
 #计算sn
    sn = (hashlib.md5(parse.quote_plus(rawStr).encode("utf8")).hexdigest())
 #由于URL里面含有中文，所以需要用parse.quote进行处理，然后返回最终可调用的url
    url = parse.quote("http://api.map.baidu.com"+queryStr+"&sn="+sn, safe="/:=&?#+!$,;'@()*[]")
    response = urlopen(url).read().decode('utf-8')
    #将返回的数据转化成json格式
    responseJson = json.loads(response)
    #获取经纬度
    lng = responseJson.get('result')['location']['lng']
    lat = responseJson.get('result')['location']['lat']
    # return lng,lat
    #print(lng,lat)
    return float(lng),float(lat)

#将文本文件的数据进行聚类并画出结果
import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=8):
    datList = []
    arr = readFlies('ChinaPlace.txt')
    for i in range(len(arr)):
        longitude, latitude = get_urt(arr[i])
        datList.append([longitude,latitude])
    # for line in open('lal.txt').readlines():
    #     lineArr = line.split('\t')
    #     datList.append([float(lineArr[0]),float(lineArr[1])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('chinaditu.png') #基于图像创建矩阵
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

#test
from numpy import *
#导入文件
#dataMat = mat(loadDataSet('testSet.txt'))
#得到数据第一列和第二列中的最大/小值
# print("min0=\n",min(dataMat[:,0]))
# print("min1=\n",min(dataMat[:,1]))
# print("max0=\n",max(dataMat[:,0]))
# print("max1=\n",max(dataMat[:,1]))
# print('k=\n',randCent(dataMat,2))
#k-均值算法求得质心
# myCentrols, clustAssing = kMeans(dataMat,4)
# print('myCentrols= \n',myCentrols)
# print('clustAssing= \n',clustAssing)
#二分k-均值算法求得质心
# dataMat = mat(loadDataSet('testSet2.txt'))
# centList, myNewAssments = biKmeans(dataMat,3)
# print('centList= \n',centList)
# print('myNewAssments=\n',myNewAssments)
#地图上点的聚类
clusterClubs(8)