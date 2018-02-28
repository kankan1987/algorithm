#!/usr/bin/python
#coding:UTF-8
import numpy
from numpy import loadtxt, where, e, reshape, transpose, log, zeros
from pandas import cut
from pylab import scatter, show, legend, xlabel, ylabel, plt, np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, Normalizer
import time

#定义数据文件路径
#file_path = '/Users/handshank/Documents/python_code/test_code/data1.txt'

#file_path = '/Users/handshank/Documents/python_code/test_code/data2.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/test0.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/test1.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/test2.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/black_data.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/data_new_201711_temp.txt'
#file_path = '/Users/handshank/Documents/python_code/test_code/201711_bw.txt'
from excel_tool import excel_tool
from user_tool import user_tool

file_path = '/Users/handshank/Documents/python_code/test_code/r_2017_12_01.txt'
#check_file_path = '/Users/handshank/Documents/python_code/test_code/black_check_data.txt'
#check_file_path = '/Users/handshank/Documents/python_code/test_code/one_month_data.txt'
#check_file_path = '/Users/handshank/Documents/python_code/test_code/data_new_201801.txt'
#check_file_path = '/Users/handshank/Documents/python_code/test_code/201712_bw.txt'
check_file_path = '/Users/handshank/Documents/python_code/test_code/r_2017_12_15.txt'

#data_path = '/Users/handshank/Documents/python_code/test_code/data_sample'
#data_path = '/Users/handshank/Documents/python_code/test_code/test_data_20180207'
#data_path = '/Users/handshank/Documents/python_code/test_code/learn_data_20180213'
#data_path = '/Users/handshank/Documents/python_code/test_code/yb_20180226'
#data_path = 'yb_20180225'
data_path = 'yb_20180220'

# 对X,y的散列绘图
#获取样本数据
def plotData(data,label_x, label_y, label_pos, label_neg, axes=None, is_show=True):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    #如果第3列为0则赋值给neg
    neg = data[:, 2] == 0
    #如果第3列为1则赋值给pos
    pos = data[:, 2] == 1
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=8, linewidth=6, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=8, label=label_neg)
    #显示x轴
    axes.set_xlabel(label_x)
    #显示y轴
    axes.set_ylabel(label_y)
    #显示标识位置
    axes.legend(frameon=True, fancybox=True);
    #axes.legend(loc='center left',bbox_to_anchor=(0.2,1.12),ncol=3)
    if is_show:
        show()

#画决策边界线
def draw_line(poly,X,param,accuracy,axes,lambdas,i):
    # 画出决策边界
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    # 间隔采样默认50个
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    # print('xx=',xx1)
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(param))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), lambdas))

#加载数据文件
def loaddata(file, delimeter):
    #加载文件，以delimeter进行分割成行和列
    data = np.loadtxt(file, delimiter=delimeter)
    #输出处理后数据的行和列
    #print('Dimensions: ',data.shape)
    #输出第2行到第6行
    #print(data[1:6,:])
    return(data)

#定义sigmoid函数
def sigmoid(z):
    #e的-z次方
    # g(z) = 1/(1+e^(-z))
    #z=W^tX
    return(1.0 / (1 + np.exp(-z)))

#定义损失函数
def costFunction(theta,X,y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0*(1.0*m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    #print("cost function value:",J)
    if np.isnan(J[0]):
        return (np.iinfo)
    return J[0]
#求偏导-梯度
def gradient(theta,X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    grad = (1.0/m)*X.dot(h-y)
    #print("grad function value:",grad)
    return grad.flatten()

#多阶正则化损失函数
def costFunctionReg(theta,reg,XX,y):
    m = y.size
    h = sigmoid(XX.dot(theta))
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))

    #print("cost reg function value:", J)
    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])

#求偏导-梯度
def gradientReg(theta,reg,XX,y):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))
    grad = (1.0/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
    #print("grad reg function value:",grad)
    return grad.flatten()

# 预测函数
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

def accuracy_data_file(file,param):
    data = loaddata(file, ',')
    y = np.c_[data[:, 9]]
    X = data[:, 1:9]
    user_data = data[:,0:1]
    #X = X / 10.0
    #X = Normalizer().fit_transform(X)
    poly = PolynomialFeatures(1)
    data = np.c_[X,y]
    XX = poly.fit_transform(X)
    #pre_value = predict(param, XX);

    # 预测值是否正确
    pre_value = predict(param, XX);
    print ("**************************")
    res = (pre_value == y.ravel())
    count = 0
    isblacks = {}
    hnUserList = []
    for i in range(0,len(res)):
        if res[i] == False:
            print ('%s=%s'%(user_data[i,0],data[i,8]))
            hnUserList.append(str(int(user_data[i,0])))
            count += 1
            isblacks[str(int(user_data[i,0]))] = data[i,8]
    print ('check fail count:%s'%count)
    # 预测准确率
    accuracy = 100.0 * sum(pre_value == y.ravel()) / y.size

    print ("**************************")
    print ("accuracy:")
    print (accuracy)
    print ("**************************")

    userTool = user_tool()
    userInfoData = userTool.getUserInfo(hnUserList,isblacks)
    print userInfoData

    excelTool = excel_tool()
    excelTool.createExcel(userInfoData['col_name'], userInfoData['data'])

def process_data_file(file):
    data = loaddata(file,',')
    #data = data/100.0
    # 获取第3列
    y = np.c_[data[:, 9]]
    # 获取第1、2列
    X = data[:, 1:9]
    #X = X/10.0

    #6阶多项式
    poly = PolynomialFeatures(1)
    #标准化数据，保证每个维度的特征数据方差为1，均值为0
    data = np.c_[X,y]
    #X = Normalizer().fit_transform(X)
    XX = poly.fit_transform(X)
    #初始化参数为0
    initial_theta = np.zeros(XX.shape[1])
    #获取图句柄
    #fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

    # 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
    # Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
    # Lambda = 1 : 这才是正确的打开方式
    # Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
    lambdas = [1]
    for i, C in enumerate(lambdas):
        #300次迭代, 'disp': True
        res = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg,options={'maxiter': 2000})
        #预测值是否正确
        pre_value = predict(res.x, XX);
        #预测准确率
        accuracy = 100.0 * sum(pre_value == y.ravel()) / y.size
        #画数据分布图
        #print ('draw dot >>>>>>>>>>>>>>>>')
        #plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i], False)
        #print ('draw line >>>>>>>>>>>>>>>>')
        #画函数曲线图
        #draw_line(poly, X, res.x, accuracy, axes, C, i)

        print('===========================================')
        print('lamba:')
        print(C)
        print('accuracy:')
        print(accuracy)
        print('param:')
        print(res.x)
        print('============================================')
        return res.x

    #显示图形
    #show()


def accuracy_data(data,param,order=2):
    cols = len(data[0])
    y = np.c_[data[:, cols-1]]
    X = data[:, 0:cols-1]
    poly = PolynomialFeatures(order)
    XX = poly.fit_transform(X)

    # 预测值是否正确
    pre_value = predict(param, XX);
    check_res = (pre_value == y.ravel())
    print ("check_res=%s"%check_res)
    # 预测准确率
    accuracy = 100.0 * sum(pre_value == y.ravel()) / y.size

    print ("**************************")
    print ("accuracy:")
    print (accuracy)
    print ("**************************")
    return accuracy

def process_data(data,order=2,count=2000):

    #6阶多项式
    poly = PolynomialFeatures(order)
    #标准化数据，保证每个维度的特征数据方差为1，均值为0
    #X = Normalizer().fit_transform(X)
    cols = len(data[0])
    #获取特征向量
    X = data[:,0:cols-1]
    y = np.c_[data[:,cols-1]]

    XX = poly.fit_transform(X)
    #初始化参数为0
    initial_theta = np.zeros(XX.shape[1])
    #获取图句柄
    #fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

    # 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
    # Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
    # Lambda = 1 : 这才是正确的打开方式
    # Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
    lambdas = [1]
    for i, C in enumerate(lambdas):
        #300次迭代, 'disp': True
        res = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg,options={'maxiter': 2000})
        pre_value = predict(res.x, XX);
        accuracy = 100.0 * sum(pre_value == y.ravel()) / y.size
        print('===========================================')
        print('lamba:')
        print(C)
        print('accuracy:')
        print(accuracy)
        print('param:')
        print(res.x)
        print('============================================')
        return res.x,accuracy


#按要求获取数据
def get_data(cols,rows,file):
    data = loaddata(file, ',')
    print ('data size:%s'%len(data))
    if cols is None:
        cols = 0
    if rows is None:
        rows = 0
    data = numpy.array(data)
    result = data[rows:,cols:]
    #print ("cols=%s"%cols)
    #print ("rows=%s"%rows)
    # if cols is None:
    #     cols = range(len(data[0]))
    #
    # if rows is None:
    #     rows = range(len(data))
    #
    # rows_len = len(rows)
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
    return result

#训练样本与实际正确率
def sample_train(data,count=1,test_size=0.4,order=1,random_state=0):
    res_data = np.zeros((count,4))
    cols = len(data[0])
    param_values = {}
    count_value = np.arange(count)
    min_value = 0
    res_param = None
    for i in range(count):
        y = data[:,cols-1]
        data_train,data_check,y_train,y_check = train_test_split(data,y,test_size=test_size,random_state=i)
        print ("this is the [%d] learnning:"%i)
        print ("sample_count:%d"%len(data_train))
        print ("check_count:%d"%len(data_check))
        #训练参数与样本准确率
        param, accuracy1 = process_data(data_train, 1)
        #校验数据准确率
        accuracy2 = accuracy_data(data_check, param, 1)
        res_data[i,0] = accuracy1
        res_data[i,1] = accuracy2
        res_data[i,2] = i
        res_data[i,3] = abs(accuracy1-accuracy2)
        #print("=====%s"%param)
        #param = np.c_[param,count_value[i-1]]
        #param_values[i] = param
        if i == 0:
            min_value = res_data[i,3]
            res_param = param
        if res_data[i,3] < min_value:
            min_value = res_data[i,3]
            res_param = param
    return res_data,res_param

def plot_data(data,label_x, label_y):
    axes = plt.gca()
    axes.scatter(data[:,2], data[:,0], marker='+', c='k', s=8)
    axes.scatter(data[:,2], data[:,1],c='y', s=8)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True);
    show()

def draw(x,y,x_text,y_text,title):
    plt.figure(figsize=(30,5))
    plt.plot(x,y,color='red',label='data_check_result')
    for i in range(1,len(x)):
        plt.text(x[i],y[i],str((x[i],round(y[i],4))))
    #plt.text(x,y,(x,y),color='red')
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    pic = time.strftime("%Y-%m-%d_%H_%S_%M",time.localtime())+ ".pdf"

    plt.savefig(pic)
    plt.show()
    #plt.savefig("xxx.jpg")

#主函数
if __name__ == "__main__":
    # param = process_data(file_path)
    # print ("####################")
    # print (param)
    # print ("####################")
    # accuracy_data(check_file_path,param)
    #cols = [2,4,5,7,8,9,10,11,12,13,25]
    col_names = ['constant','mobile_virtual','app_correlation','supply_different','supply_risk',
            'mobile_risk_area','supply_risk_area','mobile_supply','percard','rvicard','activeFrequency','im_send','im_revice']
    cols = 2
    #rows = range(1468)
    rows = None
    data = get_data(cols,rows,data_path)
    print ("the first data is (%s)"%data[0])

    res,res_param = sample_train(data, 30,0.6)
    #plot_data(res,'x=accuracy','y=count')
    draw(res[:,2],res[:,3],'count','check_value','the check result')
    print("++++++++++++++++++++++++++++++++++++")

    print(res_param)

    result = {}
    for i in range(len(res_param)):
        #result[round(res_param[i],10))]
        result[col_names[i]] = round(res_param[i],11)
        #print("%s:%s"%(col_names[i],round(res_param[i],11)))

    result = sorted(result.items(), key=lambda item: item[1], reverse=True)
    for r in result:
        print("%s:%s" % (r[0], r[1]))











