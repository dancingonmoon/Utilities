# -*- coding: utf-8 -*-
import numpy as np
import pandas
import pandas as pd
import configparser
from typing import Union
import os
from Translation_API import MStranslation_dynamicDictionary_API, BaiduTranslateAPI, BaiduTranslateAPI_domain, \
    Qwen_MT_func, Term
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


# 定义函数:DataFrame单列遍历,调用百度翻译API,并生成翻译后的DataFrame
def DF_Translate_baidu(
        source_df: pandas.DataFrame, appid, appkey, from_lang: str = "auto", to_lang: str = "en"
):
    """
    百度翻译API下的单列文本翻译, 输入DataFrame列,翻译后,返回DataFrame列,包括列名翻译;空白文本自动跳过;
    DFColum: DataFrame block
    from_lang: 源语言; 'auto'时,自动检测语言
    to_lang:目标语言
    """
    translate_df = source_df.copy()
    for source_col_name in source_df.columns:
        translate_col_list = []
        # DataFrame每列转换成列表:
        source_col_list = source_df[source_col_name].tolist()
        source_col_list.append(source_col_name)  # 将列名添加到列表末尾
        # 替换 NaN、Infinity 和 -Infinity 为 None 或其他默认值
        for x in source_col_list:
            if isinstance(x, (int, float)):
                if np.isinf(x) or np.isnan(x):
                    translate_col_list.append('')
                else:
                    translate_col_list.append(
                        BaiduTranslateAPI(str(x), appid, appkey, from_lang, to_lang)
                    )
            else:
                translate_col_list.append(
                    BaiduTranslateAPI(x, appid, appkey, from_lang, to_lang)
                )

        translate_col_name = translate_col_list.pop()  # 将列表中的最后一个元素弹出作为列名
        translate_df[translate_col_name] = translate_col_list
        translate_df.drop(columns=source_col_name, inplace=True)
    return translate_df


def DF_Translate_MS(
        source_df,
        ms_subscribeKey,
        dynamic_dict: dict = False,
        from_lang: str = None,
        to_lang: str = "en",
):
    """
    微软翻译API下的文本翻译, 输入DataFrame块,翻译后,返回DataFrame块,包括列名翻译;空白文本未自动跳过;
    source_df: 待翻译原文的DataFrame块
    from_lang: 源语言; 当from_lang 不包含(None)时, 微软翻译器自动检测语言; (注:'auto'不支持)
    to_lang: 目标语言
    """
    translate_df = source_df.copy()
    # DataFrame可能为多列：
    for source_col_name in source_df.columns:
        # DataFrame每列转换成列表:
        source_col_list = source_df[source_col_name].tolist()
        source_col_list.append(source_col_name)  # 将列名添加到列表末尾
        # 替换 NaN、Infinity 和 -Infinity 为 None 或其他默认值
        for n, x in enumerate(source_col_list):
            if isinstance(x, (int, float)):
                if np.isinf(x) or np.isnan(x):
                    source_col_list[n] = ''
                else:
                    source_col_list[n] = str(x)

        translate_col_list, response = MStranslation_dynamicDictionary_API(
            source_col_list,
            dynamic_dict=dynamic_dict,
            lang_in=from_lang,
            lang_out=to_lang,
            subscription_key=ms_subscribeKey,
        )
        if "error" in response:
            return response

        translate_col_name = translate_col_list.pop()  # 将列表中的最后一个元素弹出作为列名
        translate_df[translate_col_name] = translate_col_list
        translate_df.drop(columns=source_col_name, inplace=True)
    return translate_df


def Write2Excel(
        dataframe: pandas.DataFrame, Write2Path, Write2Sheet: str, Write2Row: int = 0, Write2Col: int | list[int] = 0,
        Writeheader: bool = True
):
    """
    将DataFrame追加写入已经存在的Excel,写入指定的sheet, 从指定的行列号开始写完整的DataFrame块;
    注: Dataframe的列数必须与Write2Col的列表长度一致
    :param dataframe: 写入的DataFrame
    :param Write2Path: 已经存在的Excel的完整路径;
    :param Write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    :param Write2Row: 写入的起始行,默认为0;
    :param Write2Col: 写入的起始列,默认为0; int|list
    :param Writeheader: bool类型, 是否写入标题行,默认为True;
    """
    if isinstance(Write2Col, int):
        Write2Col = [Write2Col]
    n_rows, n_cols = dataframe.shape
    assert len(Write2Col) == n_cols, f"Write2Col的长度{len(Write2Col)},与DataFrame的列数{n_cols}，必须相等"

    with pd.ExcelWriter(
            Write2Path, mode="a", engine="openpyxl", if_sheet_exists="overlay"
    ) as writer:
        for col_name, col in zip(dataframe.columns, Write2Col):  # 每列循环写入指定的列号
            dataframe[col_name].to_excel(
                excel_writer=writer,
                sheet_name=Write2Sheet,
                startrow=Write2Row,
                startcol=col,
                index=False,
                header=Writeheader,
            )


def Excel_block_Translate(
        translate_engine: translate_engine = 1,
        ms_subscribeKey=None,
        baidu_appid=None,
        baidu_appkey=None,
        dynamic_dict: dict = False,
        from_lang: Union[str, None] = "auto",
        to_lang="en",
        ExcelPath: str = None,
        readSheet: str = None,
        readHeader: int = None,
        readStartRow: Union[int, None] = 1,
        readCol: Union[int, list[int]] = None,
        nrows: int = None,
        write2Sheet: str = None,
        write2Row: int = 2,
        write2Col: Union[int, list[int]] = 1,
):
    """
    Excel中读取单列,或者block,并调用百度翻译API,翻译后,写入已经存在的Excel,写入指定的sheet, 从指定的行列号追加写入完整的列或者块;
    当readCol为列表时，readCol与write2Col的长度必须一致，读写列一一对应
    :param translate_engine: 枚举变量: 缺省为microsoft翻译引擎(1), 其次为baidu翻译引擎(2)
    :param ms_subscribeKey: 微软翻译API subscription key
    :param baidu_appid: 百度openAPI baidu_appid
    :param baidu_appkey: 百度openAPI baidu_appkey
    :param from_lang: 源语言
    :param to_lang: 目标语言(For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21)
    :param ExcelPath: 待读取,翻译以及写入的Excel的完整路径;
    :param readSheet: 待读取的sheet name;
    :param readHeader: 待读取标题行行号(Excel的行号,从1开始计数); 缺省None
    :param readStartRow: 待读取的起始行(Excel的行号,从1开始计数); 缺省1;
            当readHeader不为None时,readStartRow=None表示从readHeader以下开始
    :param readCol: int|list[int],待读取的起始列号或者列号的列表;(Excel的列号,从1开始计数)
    :param nrows: 待读取行的数量; 初始值None,表示读取整列;
    :param write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    :param write2Row: 写入的起始行,(不包括标题列);默认为2;
    :param write2Col: int|list[int],写入的起始列,默认为1,(Excel列号,从1开始计数)
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
        TransData = DF_Translate_baidu(
            ExcelData, baidu_appid, baidu_appkey, from_lang, to_lang
        )
    elif translate_engine == translate_engine.microsoft:
        TransData = DF_Translate_MS(
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
        dataframe=TransData,
        Write2Path=ExcelPath,
        Write2Sheet=write2Sheet,
        Write2Row=write2Row,
        Write2Col=[x - 1 for x in write2Col],
        Writeheader=writeheader
    )  # 列号从0开始计数与从1开始计数差1
    print(f"翻译并写入完成! 请查看{ExcelPath}.{write2Sheet}")


if __name__ == "__main__":
    # 读取需要翻译的Excel文件,装入DataFrame
    from_lang_ms = "zh-Hans"  # 百度: 'auto', 微软: None; 自动辨识语言;当双语混杂时,需要指定语言,否则会有漏译情况;
    from_lang_bd = "zh"  # 百度: 'auto', 微软: None; 自动辨识语言;当双语混杂时,需要指定语言,否则会有漏译情况;
    to_lang = "en"  # 百度:( 中文: zh;文言文: wyw;日本: jp; --> 伊朗语: ir; 波斯语);微软: "zh-Hans"中文简体
    ExcelPath = r"E:/Working Documents/Eastcom/Russia/Igor/专网/CTK/ТЗ по TETRA_配置_250707.xlsx"
    readSheet = "8载波"
    readHeader = 1
    readStartRow = None
    readCol = [2, 3, 4]  # 也可以用列表读取多列
    nrows = 4
    write2Sheet = "test"
    write2Row = 2
    write2Col = [2, 3, 4]

    config_path = r"e:/Python_WorkSpace/config/baidu_OpenAPI.ini"
    MSconfig_path = r"e:/Python_WorkSpace/config/Azure_Resources.ini"

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
                    "东方通信": "Eastcom",
                    "集群系统": "trunking system",
                    "调度系统": "dispatching system",
                    "调度": "dispatching", }

    # # 百度翻译引擎:
    # Excel_block_Translate(translate_engine=translate_engine.baidu, baidu_appid=baidu_appid, baidu_appkey=baidu_appkey,
    #                      from_lang=from_lang_bd, to_lang=to_lang, ExcelPath=ExcelPath,
    #                      readSheet=readSheet, readHeader=readHeader, readStartRow=readStartRow, readCol=readCol,
    #                      nrows=nrows,
    #                      write2Sheet=write2Sheet, write2Row=write2Row, write2Col=write2Col )
    # # 微软翻译引擎:
    # Excel_block_Translate(translate_engine=translate_engine.microsoft, ms_subscribeKey=ms_subscribeKey,
    #                      dynamic_dict=dynamic_dict,
    #                      baidu_appid=baidu_appid,
    #                      baidu_appkey=baidu_appkey, from_lang=from_lang_ms, to_lang=to_lang,
    #                      ExcelPath=ExcelPath,
    #                      readSheet=readSheet, readHeader=readHeader, readStartRow=readStartRow, readCol=readCol,
    #                      nrows=nrows,
    #                      write2Sheet=write2Sheet, write2Row=write2Row, write2Col=write2Col)

    # 从效果上看, 百度翻译比微软翻译速度慢很多(因为百度每个字符串调用，而MS列表输入，一次调用),但是效果似乎在列表类中文文本上,似乎比微软要好一些;可能百度在理解中文上更胜一筹.

    # print("翻译完成!")
    # 翻译句子, 从效果上看，MS明显优于百度翻译,似乎也略微优于Qwen_MT
    txt = """
    具有群组呼叫、紧急呼叫等快速指挥调度能力；支持用户优先级的设置；
    通过动态重组实现灵活的多级别分组调度指挥功能；集语音、数据、图像和视频的多业务传输能力；
    支持远程升级及版本回退；支持终端APP通过二维码扫描下载
    """
    # 1 baidu :
    response = BaiduTranslateAPI_domain(txt, baidu_appid, baidu_appkey,
                             fromLang="auto", toLang="en",
                             domain='it')
    print(f'百度domain翻译结果:{response}')
    # 2 MicroSoft:
    translate, response = MStranslation_dynamicDictionary_API(txt, lang_in=None, lang_out="en",subscription_key=ms_subscribeKey)
    print(f'MS翻译结果:{translate}')
    print(f'MS翻译score:{response[0]['detectedLanguage']['score']}')
    # 3 QwenMT:
    domains = "you are working in the field of wireless communication filed, with strong knowledge of mobile network"
    terms = []
    for k, v in dynamic_dict.items():
        terms.append({"source": k, "target": v})

    QwenMT_translate = Qwen_MT_func(txt,
                                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                                    source_lang="auto",
                                    target_lang="English",
                                    terms=terms,
                                    domains=domains)
    print(f'QwenMT翻译结果:{QwenMT_translate}')


