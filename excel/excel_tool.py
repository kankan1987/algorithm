#!/usr/bin/python
#coding:UTF-8
import time

import numpy
import openpyxl
from openpyxl import Workbook

file_name = '/Users/handshank/Documents/python_code/test_code/'

class excel_tool:
    #初始化
    def __init__(self):
        pass;
    def createExcel(self,columnNames,data,path=''):
        wb = Workbook()
        ws = wb.active
        ws.append(columnNames)
        rows_len = len(data)
        for d in data:
            if isinstance(d,numpy.ndarray):
                ws.append(d.tolist())
            else:
                ws.append(d)
        #保存表格，并命名为‘xxxx’户.xls
        excelTime = time.strftime("%Y%m%d_%H%M%S")
        name = excelTime + "_运算数据.xls"
        file_path = path + name
        wb.save(file_path)

        return file_path,name