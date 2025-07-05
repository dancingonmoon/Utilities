# -*- coding: utf-8 -*-
import numpy as np
import requests
import random
from hashlib import md5
import pandas as pd
import configparser
from typing import Union
import os
from Translation_API import MStranslation_API, MStranslation_dynamicDictionary_API
from enum import Enum


class translate_engine(Enum):
    microsoft = 1
    baidu = 2


# https://fanyi-api.baidu.com/api/trans/product/desktop?req=developer


def config_read(
    config_path, section="DingTalkAPP_chatGLM", option1="Client_ID", option2=None
):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


# Generate salt and sign
def make_md5(s, encoding="utf-8"):
    return md5(s.encode(encoding)).hexdigest()


# 定义百度翻译ＡＰＩ函数
def BaiduTranslateAPI(text, appid, appkey, from_lang="auto", to_lang="en"):
    """
    baidu_appid: 百度openAPI baidu_appid
    baidu_appkey: 百度openAPI baidu_appkey
    For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21
    """
    salt = random.randint(32768, 65536)
    s = "".join([appid, text, str(salt), appkey])
    sign = make_md5(s)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "appid": appid,
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "salt": salt,
        "sign": sign,
    }
    endpoint = "http://api.fanyi.baidu.com"
    path = "/api/trans/vip/translate"
    url = "".join([endpoint, path])
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    try:
        trans_result = result["trans_result"][0]["dst"]
        return trans_result
    except Exception as e:
        return result

    # 第一个方括号为字典的指定key取值,第二个方括号为之前取出的值为列表,列表的第一个元素,
    # 第三个方括号为取出的第一个元素继续为字典,取字典的值


# 定义函数:DataFrame单列遍历,调用百度翻译API,并生成翻译后的DataFrame
def DFColumnTranslate_baidu(
    DFColumn, appid, appkey, from_lang: str = "auto", to_lang: str = "en"
):
    """
    百度翻译API下的单列文本翻译, 输入DataFrame列,翻译后,返回DataFrame列,包括列名翻译;空白文本自动跳过;
    DFColum: DataFrame单列
    from_lang: 源语言; 'auto'时,自动检测语言
    to_lang:目标语言
    """
    Translate = []
    for value in DFColumn.values:
        x = value[0]
        if x != "":
            if isinstance(x, (int, float)):
                if np.isinf(x) or np.isnan(x):
                    Translate.append('')
                else:
                    Translate.append(
                BaiduTranslateAPI(str(x), appid, appkey, from_lang, to_lang)
            )

            else:
                Translate.append(
                BaiduTranslateAPI(x, appid, appkey, from_lang, to_lang)
            )

        else:
            Translate.append(value)
    Column_name = DFColumn.keys()[0]
    if isinstance(Column_name, str):
        Column_name = BaiduTranslateAPI(Column_name, appid, appkey, from_lang, to_lang)

    Translate = pd.DataFrame(Translate, columns=[Column_name])
    return Translate


def DFColumnTranslate_MS(
    DFColumn,
    ms_subscribeKey,
    dynamic_dict: dict = False,
    from_lang: str = None,
    to_lang: str = "en",
):
    """
    微软翻译API下的单列文本翻译, 输入DataFrame列,翻译后,返回DataFrame列,包括列名翻译;空白文本自动跳过;
    DFColum: DataFrame单列
    from_lang: 源语言; 当from_lang 不包含时, 微软翻译器自动检测语言; (注:'auto'不支持)
    to_lang:目标语言
    """
    # DataFrame转换成列表:
    DFColumn_list = DFColumn.values.tolist()
    # 替换 NaN、Infinity 和 -Infinity 为 None 或其他默认值
    for n, x in enumerate(DFColumn_list):
        x = x[0]
        DFColumn_list[n] = x
        if isinstance(x, (int,float)):
            if np.isinf(x) or np.isnan(x):
                DFColumn_list[n] = ''
            else:
                DFColumn_list[n] = str(x)

    Translate, response = MStranslation_dynamicDictionary_API(
        DFColumn_list,
        dynamic_dict=dynamic_dict,
        lang_in=from_lang,
        lang_out=to_lang,
        subscription_key=ms_subscribeKey,
    )
    if "error" in response:
        return response

    Column_name = DFColumn.keys()[0]
    if isinstance(Column_name, str):
        Column_name, response = MStranslation_dynamicDictionary_API(
            Column_name,
            dynamic_dict=dynamic_dict,
            lang_in=from_lang,
            lang_out=to_lang,
            subscription_key=ms_subscribeKey,
        )
        if "error" in response:
            return response

    Translate = pd.DataFrame(Translate, columns=[Column_name])
    return Translate


def Write2Excel(
    DFColumn, Write2Path, Write2Sheet, Write2Row=0, Write2Col=0, Writeheader=True
):
    """
    将DataFrame追加写入已经存在的Excel,写入指定的sheet, 从指定的行列号开始写完整的DataFrame块;
    DataFrame: 写入的DataFrame
    Write2Path: 已经存在的Excel的完整路径;
    Write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    Write2Row: 写入的起始行,默认为0;
    Write2Col: 写入的起始列,默认为0,
    Writeheader: bool类型, 是否写入标题行,默认为True;
    """
    with pd.ExcelWriter(
        Write2Path, mode="a", engine="openpyxl", if_sheet_exists="overlay"
    ) as writer:
        DFColumn.to_excel(
            excel_writer=writer,
            sheet_name=Write2Sheet,
            startrow=Write2Row,
            startcol=Write2Col,
            index=False,
            header=Writeheader,
        )


def ExcelColumnTranslate(
    translate_engine: translate_engine = 1,
    ms_subscribeKey=None,
    baidu_appid=None,
    baidu_appkey=None,
    dynamic_dict: dict = False,
    from_lang: Union[str, None] = "auto",
    to_lang="en",
    ExcelPath: str = None,
    readSheet:str=None,
    readHeader: int = None,
    readStartRow: Union[int, None] = 1,
    readCol: Union[int, list[int]] = None,
    nrows: int = None,
    write2Sheet: str = None,
    write2Row: int = 2,
    write2Col: int = 1,
):
    """
    Excel中读取单列,或者block,并调用百度翻译API,翻译后,写入已经存在的Excel,写入指定的sheet, 从指定的行列号追加写入完整的列或者块;
    translate_engine: 枚举变量: 缺省为microsoft翻译引擎(1), 其次为baidu翻译引擎(2)
    ms_subscribeKey: 微软翻译API subscription key
    baidu_appid: 百度openAPI baidu_appid
    baidu_appkey: 百度openAPI baidu_appkey
    from_lang: 源语言
    to_lang: 目标语言(For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21)
    ExcelPath: 待读取,翻译以及写入的Excel的完整路径;
    readSheet: 待读取的sheet name;
    readHeader: 待读取标题行行号(Excel的行号,从1开始计数); 缺省None
    readStartRow: 待读取的起始行(Excel的行号,从1开始计数); 缺省1;
        当readHeader不为None时,readStartRow=None表示从readHeader以下开始
    readCol: 待读取的起始列号或者列号的列表;(Excel的列号,从1开始计数)
    nrows: 待读取行的数量; 初始值None,表示读取整列;
    write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    write2Row: 写入的起始行,(不包括标题列);默认为2;
    write2Col: 写入的起始列,默认为1,(Excel列号,从0开始计数)
    """
    # 检查文件是否存在，并且具有写权限
    if not os.path.exists(ExcelPath):
        print(f"{ExcelPath},文件不存在")
        return
    if not os.access(ExcelPath, os.W_OK):
        print(f"{ExcelPath},无写权限.")
        return
    # 检查xlsx文件是否已经有临时文件打开,从而判断是否EXCEL文件已经打开:
    dirname, filename = os.path.split(ExcelPath)
    filename = "~$".join(["", filename])
    temp_path = os.path.join(dirname, filename)
    if os.path.exists(temp_path):
        print(f"{ExcelPath},文件已经打开,禁止写入.")
        return

    if readHeader is not None:
        if readStartRow is None:
            skiprows = None
        else:
            if readStartRow > readHeader:
                skiprows = range(readHeader, readStartRow - 1)
            else:
                skiprows = None

        readHeader -= 1  # 因为读取的标题行行号是从1开始计数的,所以要减1,变成index
        writeheader = True
        write2Row -= 2  # 带标题,至少从第二行开始写入,因为第一行要留标题
    else:
        skiprows = range(0, readStartRow - 1)
        writeheader = False
        write2Row -= 1

    # 列号转换成Excel计数,从1开始
    if isinstance(readCol, list):
        readCol = [c - 1 for c in readCol]
    else:
        readCol = [readCol - 1]

    ExcelData = pd.read_excel(
        ExcelPath,
        readSheet,
        header=readHeader,
        skiprows=skiprows,
        usecols=readCol,
        nrows=nrows,
        na_values="",
    )
    if translate_engine == translate_engine.baidu:
        TransData = DFColumnTranslate_baidu(
            ExcelData, baidu_appid, baidu_appkey, from_lang, to_lang
        )
    elif translate_engine == translate_engine.microsoft:
        TransData = DFColumnTranslate_MS(
            ExcelData,
            ms_subscribeKey=ms_subscribeKey,
            from_lang=from_lang,
            to_lang=to_lang,
            dynamic_dict=dynamic_dict,
        )
        if "error" in TransData:
            print(TransData["error"])
            return
    else:
        print("translate_engine参数错误")
        return

    Write2Excel(
        TransData, ExcelPath, write2Sheet, write2Row, write2Col - 1, writeheader
    )  # -2 因为index与行号差1
    print(f"翻译并写入完成! 请查看{ExcelPath}.{write2Sheet}")


if __name__ == "__main__":
    # 读取需要翻译的Excel文件,装入DataFrame
    from_lang_ms = "zh-Hans"  # 百度: 'auto', 微软: None; 自动辨识语言;当双语混杂时,需要指定语言,否则会有漏译情况;
    from_lang_bd = "zh"  # 百度: 'auto', 微软: None; 自动辨识语言;当双语混杂时,需要指定语言,否则会有漏译情况;
    to_lang = "en"  # 百度:( 中文: zh;文言文: wyw;日本: jp; --> 伊朗语: ir; 波斯语);微软: "zh-Hans"中文简体
    ExcelPath = r"L:/temp\Eastcom/小型化TETRA集群通信项目系统配置及报价_250304.xlsx"
    readSheet = "三基站+交换中心250122"
    readHeader = 2
    readStartRow = None
    readCol = 3  # 也可以用列表读取多列
    nrows = 45
    write2Sheet = "Sheet3"
    write2Row = 2
    write2Col = 1

    config_path = r"l:/Python_WorkSpace/config/baidu_OpenAPI.ini"
    MSconfig_path = r"l:/Python_WorkSpace/config/Azure_Resources.ini"

    baidu_appid, baidu_appkey = config_read(
        config_path, section="baidu_OpenAPI", option1="appid", option2="appkey"
    )
    ms_subscribeKey = config_read(
        MSconfig_path,
        section="MS_translation",
        option1="subscription_key",
        option2=None,
    )

    dynamic_dict = {"东信": "Eastcom",
                    "东方通信": "Eastcom",}

    # # 百度翻译引擎:
    ExcelColumnTranslate(translate_engine=translate_engine.baidu, baidu_appid=baidu_appid, baidu_appkey=baidu_appkey,
                         from_lang=from_lang_bd, to_lang=to_lang, ExcelPath=ExcelPath,
                         readSheet=readSheet, readHeader=readHeader, readStartRow=readStartRow, readCol=readCol,
                         nrows=nrows,
                         write2Sheet=write2Sheet, write2Row=write2Row, write2Col=write2Col)
    # # 微软翻译引擎:
    ExcelColumnTranslate(translate_engine=translate_engine.microsoft, ms_subscribeKey=ms_subscribeKey, dynamic_dict=dynamic_dict,
                         baidu_appid=baidu_appid,
                         baidu_appkey=baidu_appkey, from_lang=from_lang_ms, to_lang=to_lang,
                         ExcelPath=ExcelPath,
                         readSheet=readSheet, readHeader=readHeader, readStartRow=readStartRow, readCol=readCol,
                         nrows=nrows,
                         write2Sheet=write2Sheet, write2Row=write2Row, write2Col=write2Col + 1)

    # 从效果上看, 百度翻译比微软翻译速度慢很多,但是效果似乎在列表类中文文本上,似乎比微软要好一些;可能百度在理解中文上更胜一筹.
    print("翻译完成!")


