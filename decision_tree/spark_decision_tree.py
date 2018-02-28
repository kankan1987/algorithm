#!/usr/bin/python
#-- coding:UTF-8 --
#
#数据分析服务，从hive中获取取用户特征数据，通过决策树校验证结果是否一致
#data analysis service, obtain the user characteristic data from hive,
# verify whether the result is consistent through the decision tree
# create by kk 2018-02-02


from __future__ import print_function

# $example on:spark_hive$
from math import log
from os.path import expanduser, join

import time
import datetime
import sys
import operator

import numpy
import numpy as np
from pyspark.shell import spark
from pyspark.sql import SparkSession

#from email_tool import email_tool

reload(sys)
sys.setdefaultencoding('utf-8')


def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：数据集的香农熵
    描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy
        if (splitInfo == 0): # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#获取数据的增益值
def getAllFeaturesInfoGain(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainV = 0.0
    bestInfoGainRatio = 0.0
    bestFeatureV = -1
    bestFeature = -1
    featureInGains = []

    for i in range(numFeatures):
        infoGains = []
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy

        infoGains.append(newEntropy)
        #信息增益值
        infoGains.append(infoGain)
        infoGains.append(splitInfo)
        if (splitInfo == 0): # fix the overflow bug
            infoGains.append(0)
            featureInGains.append(infoGains)
            continue
        #信息增益率
        infoGainRatio = infoGain / splitInfo
        infoGains.append(infoGainRatio)

        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i

        if (infoGain > bestInfoGainV):
            bestInfoGainV = infoGain
            bestFeatureV = i
        featureInGains.append(infoGains)

    return bestFeatureV,bestFeature,featureInGains

def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = None
    for key in secondDict.keys():

        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    if classLabel is None:
        print ("key = %s"%key)
        print ("testVec = %s"%testVec)
        print ("featIndex = %s"%featIndex)
        print ("featLabels = %s"%featLabels)
        print ("secondDict = %s"%secondDict)
    return classLabel

def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def storeTree(inputTree, filename):
    """
    输入：决策树，保存文件路径
    输出：
    描述：保存决策树到文件
    """
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    """
    输入：文件路径名
    输出：决策树
    描述：从文件读取决策树
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

#从hive中获取样本数据生成决策树
def createDataSet():
    dataSet = []
    labels = ['test1','test2','test3','test4','test5','test6','test7','test8','test9']
    for line in file:
        data = line.strip('\n').replace(' ','').split(',')
        dataSet.append(data[1:])
    return dataSet, labels

#从hive中获取校验数据
def getCheckSet():
    testSet = [[0, 1, 0, 0],
               [1, 2, 1, 0],
               [2, 9, 1, 0],
               [0, 1, 1, 1],
               [1, 0, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet

#get the source data
def getSourceData(time_param):
    #查询数据语句#
    sql = "select tt.hnuserid, tt.totalscore, rf2.name_deceive,rf2.user_new,rf2.mobile_virtual," \
          "rf2.app_correlation, rf2.card_different, rf2.supply_different, rf2.supply_risk," \
          "rf2.mobile_risk_area,rf2.supply_risk_area,rf2.mobile_supply,rf2.percard,rf2.rvicard,rf2.devNums," \
          "rf2.registerlen,rf2.mActives,rf2.dwActives,rf2.wActives,tt.isblack " \
          "from( select t.hnuserid,t.totalscore,case when hn.status is null then '0' else hn.status end as isblack " \
          "from ( select hnuserid, totalscore from risk_user_portrait.risk_up_featuresScore_total " \
          "where day='" + time_param + "'" \
          ")t " \
          "left outer join risk_business_source.hn_user_black hn on t.hnuserid = hn.hn_user_id " \
          "left outer join risk_business_source.hnuser hu on t.hnuserid = hu.hnuserid " \
          ") tt join( select name_deceive, user_new, mobile_virtual, app_correlation, card_different, " \
          "supply_different, supply_risk, mobile_risk_area, supply_risk_area, mobile_supply, hnuserid " \
          "from risk_user_portrait.risk_up_featuresscore_more " \
          "where day='" + time_param + "'" \
          ") rf on tt.hnuserid = rf.hnuserid " \
          "left join(select hnuserid," \
          "name_deceive, user_new, mobile_virtual, app_correlation, card_different, " \
          "supply_different, supply_risk, mobile_risk_area, supply_risk_area, mobile_supply" \
          "percard,rvicard,devNums,registerlen,mActives,dwActives,wActives " \
          "from risk_user_portrait.risk_up_featuresscore_more2 " \
          "where day='" + time_param + "'" \
          ")rf2 on tt.hnuserid = rf2.hnuserid"

    # The results of SQL queries are themselves DataFrames and support all normal functions.


    #parse the sql statement
    sqlDF = spark.sql(sql)
    #get the hive data
    rows = sqlDF.collect()
    rows_count = len(rows)

    #data conversion
    data = np.zeros((rows_count, 15))
    for i in range(rows_count):
        data[i][0] = rows[i]['hnuserid']
        data[i][1] = rows[i]['totalscore']
        data[i][2] = rows[i]['name_deceive']
        data[i][3] = rows[i]['user_new']
        data[i][4] = rows[i]['mobile_virtual']
        data[i][5] = rows[i]['app_correlation']
        data[i][6] = rows[i]['card_different']
        data[i][7] = rows[i]['supply_different']
        data[i][8] = rows[i]['supply_risk']
        data[i][9] = rows[i]['mobile_risk_area']
        data[i][10] = rows[i]['supply_risk_area']
        data[i][11] = rows[i]['mobile_supply']
        data[i][12] = rows[i]['percard']
        data[i][13] = rows[i]['rvicard']
        data[i][14] = rows[i]['isblack']
    labels = ['hnuserid','totalscore','name_deceive','user_new','mobile_virtual',
              'app_correlation','card_different','supply_different',
              'supply_risk','mobile_risk_area','supply_risk_area',
              'mobile_supply','percard','rvicard','isblack',]
    return data,labels

#加载数据文件
def loaddata(file, delimeter):
    #加载文件，以delimeter进行分割成行和列
    data = np.loadtxt(file, delimiter=delimeter)
    #输出处理后数据的行和列
    #print('Dimensions: ',data.shape)
    #输出第2行到第6行
    #print(data[1:6,:])
    return(data)

#通过文件获取数据
def getSourceDataFromFile(filePath):
    data = loaddata(filePath, ',')
    print ('data size:%s'%len(data))
    cols = None
    rows = None
    if cols is None:
        cols = 0
    if rows is None:
        rows = 0
    data = numpy.array(data)
    result = data[rows:,cols:]
    #print ("cols=%s"%cols)
    #print ("rows=%s"%rows)
    #cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,19]
    #cols = [0,1,4, 5, 7, 8, 9, 10, 11, 12, 13, 25]
    # cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # rows = None
    # if cols is None:
    #     cols = range(len(data[0]))
    #
    # if rows is None:
    #     rows = range(len(data))
    #
    # cols_len = len(cols)
    # data_row_len = len(data)
    # res_data = np.zeros((data_row_len,1))
    #
    # for i in cols:
    #     res_data = np.c_[res_data,data[:,i]]
    # res_data = res_data[:,1:]
    # res = np.zeros((1,cols_len))
    # for i in rows:
    #     res = np.r_[res,[res_data[i,:]]]
    # result = res[1:,:]

    # labels = ['hnuserid','totalscore','name_deceive','user_new','mobile_virtual',
    #           'app_correlation','card_different','supply_different',
    #           'supply_risk','mobile_risk_area','supply_risk_area',
    #           'mobile_supply','percard','rvicard','isblack']
    labels = ['hnuserid','totalscore','mobile_virtual',
              'app_correlation','supply_different',
              'supply_risk','mobile_risk_area','supply_risk_area',
              'mobile_supply','percard','rvicard','activefrequency','im_send','im_revice','isblack']
    return result,labels

#获取用户信息
def getUserInfo(user_id_list):
    if len(user_id_list) > 0:
        user_id_str = ",".join(user_id_list)
        sql = "select hn.hnuserid,hn.linkman,hnaccount.useraccount," \
          "hnaccount.mobile,hn.risklevel,hn.createtime,hnaccount.lastlogontime," \
          "hnblack.status from risk_business_source.hnuser hn " \
          "left join risk_business_source.hnuseraccount hnaccount on hn.hnuserid = hnaccount.hnuserid " \
          "left join risk_business_source.hn_user_black hnblack on hn.hnuserid = hnblack.hn_user_id " \
          "where hn.hnuserid in(" + user_id_str + ")"
        sqlDF = spark.sql(sql)
        rows = sqlDF.collect()
        rows_count = len(rows)
        result = []
        cols = ['用户ID','用户名','用户账号','用户手机号','风险等级','注册时间','最后一次登录时间','是否在黑名单']
        for i in range(rows_count):
            r = []
            r.append(rows[i]['hnuserid'])
            r.append(rows[i]['linkman'])
            r.append(rows[i]['useraccount'])
            r.append(rows[i]['mobile'])
            r.append(rows[i]['risklevel'])
            r.append(rows[i]['createtime'])
            r.append(rows[i]['lastlogontime'])
            r.append(rows[i]['status'])
            result.append(r)
        return cols,result
    else:
        print("the user list is is empty")

#获取html邮件内容
def getHtml(blackData,abnormalData,otherData,cols):
    htmlText = ''
    colLen = len(cols)
    if colLen > 0:
        htmlText += "<tr><th>结果分类</th>"
        for i in range(colLen):
            htmlText += ("<th>%s</th>"%cols[i])
        htmlText += "</tr>"

        rowLen = len(blackData)
        if rowLen > 0:
            htmlText += ("<tr><th rowspan='%s'>预测为黑名单</th>"%rowLen)
            for i in range(rowLen):
                if i > 0:
                    htmlText += "<tr>"
                for j in range(colLen):
                    htmlText += ("<td>%s</td>"%blackData[i,j])
                htmlText += "</tr>"
        rowLen = len(abnormalData)
        if rowLen > 0:
            htmlText += ("<tr><th rowspan='%s'>无法预测用户</th>"%rowLen)
            for i in range(rowLen):
                if i > 0:
                    htmlText += "<tr>"
                for j in range(colLen):
                    htmlText += ("<td>%s</td>"%blackData[i,j])
                htmlText += "</tr>"
        rowLen = len(otherData)
        if rowLen > 0:
            htmlText += ("<tr><th rowspan='%s'>预测为正常用户</th>"%rowLen)
            for i in range(rowLen):
                for j in range(colLen):
                    if i > 0:
                        htmlText += "<tr>"
                    for j in range(colLen):
                        htmlText += ("<td>%s</td>" % blackData[i, j])
                    htmlText += "</tr>"
    return htmlText

#组装有你文本内容
def getTextEmail(accuracy,nowTime,rowsCount,getDataCostTime,calCostTime,blackCount,param):
    text = "<tr><th>用户黑名预测分析报告<th></tr>" \
           "<tr><td>分析算法:[决策树算法]<td></tr>" \
           "<tr><td>准确率:[%s]<td></tr>" \
           "<tr><td>执行程序时间:[%s]</td></tr>" \
           "<tr><td>获取数据条数:[%s]条</td></tr>" \
           "<tr><td>获取数据耗时:[%s]秒</td></tr>" \
           "<tr><td>处理数据耗时:[%s]秒</td></tr>" \
           "<tr><td>预测异常数据:[%s]条</td></tr>" \
           "<tr><td>本次执行的参数:%s</td></tr>" \
           %(accuracy,nowTime,rowsCount,getDataCostTime,
             calCostTime,blackCount,param)
#预测数据
def checkData(timeStr):

    desicionTree = {}
    try:
        desicionTree = grabTree('spark_decision_tree.txt')
        print("get the desicion tree is [%s]" % desicionTree)
    except Exception:
        print("the spark_decision_tree.txt can't to read")
        return None

    #获取预测数据
    time0 = time.time()
    data,labels = getSourceData(timeStr)
    time1 = time.time()
    #获取数据花费的时间
    getDataCostTime = time1 - time0
    colLen = len(labels)
    rowLen = len(data)
    print ("get data [%s] cost time [%s]s"%(rowLen,getDataCostTime))

    if rowLen > 0:
        dataSet = data[:, 2:(colLen - 3)]
        labelSet = labels[2:(colLen - 3)]
        ret = classifyAll(desicionTree, labelSet, dataSet)
        retLen = len(ret)
        #无法预测的用户
        abnormalData = []
        #预测正常当是用户却在黑名单
        otherData = []
        #预测为黑名单当是用户却正常
        blackData = []
        #正常用户
        commData = []
        # 计算数据花费的时间
        calCostTime = time.time() - time1
        print ("calculate data cost [%s]s"%calCostTime)

        accuracy = 0.0
        if rowLen == retLen:
            for i in range(retLen):
                if ret[i] is None:
                    abnormalData.extend(dataSet[i,0])
                elif dataSet[i,0] != ret[i]:
                    if ret[i] == '0':
                        otherData.extend(dataSet[i,0])
                    elif ret[i] == '1':
                        blackData.extend(dataSet[i,0])
                else :
                    commData.extend(dataSet[i,0])

            cols0,result0 = getUserInfo(abnormalData)
            cols1,result1 = getUserInfo(otherData)
            cols2,result2 = getUserInfo(blackData)
            cols = []
            if len(cols0) > 0:
                cols = cols0
            elif len(cols1) > 0:
                cols = cols1
            elif len(cols2) > 0:
                cols = cols2
            if len(commData) > 0:
                accuracy = float(len(data))/len(commData)

            #发送邮件
            html = getHtml(blackData, abnormalData, otherData, cols)
            titleHtml = getTextEmail(accuracy,timeStr,len(data),getDataCostTime,calCostTime,len(result2),desicionTree)
            # email = email_tool('用户黑名预测分析报告(%s)' % timeStr)
            # email.set_email_address('xiekang@cnhnkj.com', ['xiekang@cnhnkj.com,usermanager@cnhnkj.com'])
            # email.set_html_body(titleHtml, html)
            # email.send()
            print ("send email success !")
            #生成excel
        else:
            print ("data calculation failed")

#生成机器学习html报告
def createLearHtml(desicionTree,rowsLen,getDataCostTime,calDataCostTime,bestFeatureValue,bestFeatureRadio,labels,infoGains):
    # "<tr><td>计算结果决策树:%s</td></tr>" \
    titlehtml = "<tr><th>黑名单预测决策树算法机器学习报告</th</tr>" \
           "<tr><td>机器学习样本数据:[%s]条</td></tr>" \
           "<tr><td>获取样本数据耗时:[%s]秒</td></tr>" \
           "<tr><td>样本数据计算耗时:[%s]秒</td></tr>" \
           "<tr><td>数据特征类型:[%s]</td></tr>" \
            "<tr><td>增益值最大特征:[%s]</td></tr>" \
            "<tr><td>增益率最大特征:[%s]</td></tr>"\
            %(rowsLen,getDataCostTime,calDataCostTime,labels,bestFeatureValue,bestFeatureRadio)

    bodyHtml = ""
    labelsLen = len(labels)
    infoGainsLen = len(infoGains)
    result = {}
    if labelsLen > 0 and labelsLen == infoGainsLen:
        bodyHtml1 = "<tr><th>特征<th>"
        bodyHtml2 = "<tr><th>熵<th>"
        bodyHtml3 = "<tr><th>增益值<th>"
        bodyHtml4 = "<tr><th>增益率<th>"
        for i in range(labelsLen):
            bodyHtml1 += ("<td>%s</td>"%labels[i])
            bodyHtml2 += ("<td>%s</td>" % '{:.20f}'.format(infoGains[i][0],20))
            bodyHtml3 += ("<td>%s</td>" % '{:.20f}'.format(infoGains[i][1],20))
            #print ('gain:%s'%'{:.20f}'.format(infoGains[i][1],20))
            result[labels[i]] = '{:.20f}'.format(infoGains[i][1],20)
            bodyHtml4 += ("<td>%s</td>" % '{:.20f}'.format(infoGains[i][3],20))
        bodyHtml1 += "</tr>"
        bodyHtml2 += "</tr>"
        bodyHtml3 += "</tr>"
        bodyHtml4 += "</tr>"
        bodyHtml = bodyHtml1 + bodyHtml2 + bodyHtml3 + bodyHtml4

    result = sorted(result.items(), key=lambda item: item[1], reverse=True)
    for r in result:
        print("%s:%s" % (r[0], r[1]))

    return titlehtml,bodyHtml

#机器学习生成决策树
def learnData(timeStr):
    #获取预测数据
    time0 = time.time()
    #data,labels = getSourceData(timeStr)
    data, labels = getSourceDataFromFile("yb_20180220")
    time1 = time.time()
    getDataCostTime = time1 - time0
    rowsLen = len(data)
    colsLen = len(labels)
    print('get the source data [%s],cost time [%s]s'%(rowsLen, getDataCostTime))

    if rowsLen > 0:
        dataSet = data[:,2:]
        print (dataSet[0])
        dataSet = dataSet.tolist()
        labelsTmp = labels[2:len(labels)-1]
        print (labelsTmp)
        #获取决策树
        desicionTree = createTree(dataSet, labelsTmp)
        print ("create the desicion tree is [%s]"%desicionTree)
        time2 = time.time()
        calDataCostTime = time2 - time1

        bestFeatureV,bestFeature,infoGains = getAllFeaturesInfoGain(dataSet)
        labelsCol = labels[2:len(labels)-1]
        bestFeatureValue = labelsCol[bestFeatureV]
        bestFeatureRadio = labelsCol[bestFeature]
        print ('labelsCol=%s'%len(labelsTmp))
        print ('infoGains=%s'%len(infoGains))
        titleHtml,html = createLearHtml(desicionTree, rowsLen, getDataCostTime, calDataCostTime, bestFeatureValue,bestFeatureRadio, labelsCol, infoGains)
        # email = email_tool('用户黑名预测分析报告(%s)' % timeStr)
        # email.set_email_address('xiekang@cnhnkj.com', ['xiekang@cnhnkj.com'])
        # email.set_html_body(titleHtml, html)
        # email.send()

        try:
            storeTree(desicionTree, 'classifierStorage.txt')
        except Exception:
            print ('save the desicionTree file classifierStorage.txt error')


#主程序
def main():
    #获取时间参数
    now = datetime.datetime.now()
    print ('the process start time is %s'%now.strftime("%Y-%m-%d %H:%S:%M"))
    yesterday = now - datetime.timedelta(days=1)
    yesterdayStr = yesterday.strftime("%Y%m%d")
    print ('the param yesterday is %s'%yesterdayStr)
    beforeYesterDay = now - datetime.timedelta(days=2)
    beforeYesterDayStr = yesterday.strftime("%Y%m%d")

    # 默认数据表的管理路径
    warehouse_location = 'spark-warehouse'
    #添加spak任务
    # spark = SparkSession \
    #     .builder \
    #     .appName("spark analysis risk data by decision tree") \
    #     .config("spark.sql.warehouse.dir", warehouse_location) \
    #     .enableHiveSupport() \
    #     .getOrCreate()

    #获取参数
    paramCount = sys.argv
    param = None
    if paramCount > 1:
        param = sys.argv[1]
        if paramCount >2:
            beforeYesterDayStr = sys.argv[2]

    if param is None or param == 'check':
        #校验数据
        print ("start to check the data,the param is %s"% yesterdayStr)
        checkData(yesterdayStr)
    elif param == 'learn':
        #机器学习生成决策树
        print("start to learn the data,the param is %s" % beforeYesterDayStr)
        learnData(beforeYesterDayStr)
    else :
        print ("####################################################")
        print ("command eg: spark2-submit spark_decision_tree.py [param] [date]")
        print ("the value of param is empty is to check the data")
        print ("the value of param is 'check' is to check the data")
        print ("the value of param is 'learn' is to create the tree")
        print ("the value of date is 'learn' is to the learn date")
        print ("to check the data eg:spark2-submit spark_decision_tree.py")
        print ("to lean the data  eg:spark2-submit spark_decision_tree.py learn")

    spark.stop()

if __name__ == '__main__':
    main()
