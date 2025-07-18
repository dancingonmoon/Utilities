import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from typing import Union, List
from mysql_handler import config_read
import os
import time
import re


def semantic_batch_search(
    client,
    query: Union[str, List[str], np.ndarray],
    sentences: Union[list, dict, pd.Series],
    embedding_txt_key: str = None,
    embedded_vec_key: str = None,
    k: int = 3,
    return_indices_samples: bool = False,
):
    """
    Faiss需要提供向量空间,以实现语义检索;query为向量时,sentences应也包括同等向量算法向量化后的向量空间;当query为文本时,使用glm的向量模型Embedding-2进行向量化后,再进行向量空间搜索.
    query可以是单一字符串或者单一向量,在sentences列中搜索与query最相似的k个句子; 也可以是一个字符串列表,或者二维向量矩阵,搜索语义最接近的k个句子.(使用huggingface的Dataset调用的faiss方法);
    1) query: Union[str,List[str], np.ndarray]; query如果是空字符串,或者包含空的字符串,会伪造一个随机(1024,)维的向量,取值范围[-1,1]
    1) sentences为一字典时,则包含待embedding的文本key=embedding_txt_key,然后字典再转换成huggingface的Dataset;也可以包括向量化后的向量空间,以key=embedded_vector_key表示;
    2) sentences为list时,则列表内包含的或是待embedding的字符串,则转换成字典,key='_embedding_txt';;
    3) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本;或者serach()获得最佳的k个dataset的样本的位置
    :param client: object; zhipuAI client (已经送入API_KEY)
    :param query: Union[str,List[str], np.ndarray]; 欲搜索的关键词或者句子,或者向量,或者字符串列表;query不应有空字符串出现
    :param sentences: 列表或者字典,送入embedding-2模型,获得长度1024的向量列表;key:列名;value:文本;不应有空字符串出现
    :param embedding_txt_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索;当sentences为list,可为None;sentence为字典时,缺省第一列可设为None
    :param embedded_vector_key: str: 如果预存有向量化后的向量空间,此处为其key名;
    :param k: int; 返回最相似的句子数量
    :param return_indices_samples: bool;是否返回检索后在序列中的位置;True: 返回检索位置;False:返回nearest_samples
    :return: scoresList[List[float], nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳),类型分别为: List[List[float], List[dict],
            或者 scores, indices, 类型: List[List[float], List[int]
    """
    sentences_txt = []
    sentences_vec = []

    # 处理sentences字典/字符串列表: 字典embedding_txt_key下的value: 1) list[str],2) list[一维向量],3) 二维向量;
    faiss_index_name = "_embedding"
    # sentences 字典提出sentences_txt
    if isinstance(sentences, dict):
        if embedding_txt_key is None:  # 未给待embedding_txt列的列名,取第一列
            embedding_txt_key = list[sentences.keys()][0]
            sentences_txt = sentences[embedding_txt_key]
        else:
            sentences_txt = sentences[embedding_txt_key]

    elif isinstance(sentences, list) and all(isinstance(x, str) for x in sentences):
        sentences_txt = sentences
        sentences = {"_embedding_txt": sentences_txt}
    elif isinstance(sentences, pd.Series):
        sentences_txt = sentences.values
        sentences = {"_embedding_txt": sentences_txt}
    elif isinstance(sentences, np.ndarray):
        sentences = {faiss_index_name: sentences_txt}

    # sentences 向量化
    if (
        embedded_vec_key is None and faiss_index_name not in sentences
    ):  # sentences字典已经包含了向量化过的向量空间,则无需再向量化
        for sentence in sentences_txt:
            if sentence == "":  # 如果是空字符,则伪造一个1024维度的向量,取值范围为-1,1之间;
                embedding_list = np.random.uniform(-1, 1, size=(1024,)).tolist()
            else:
                response = client.embeddings.create(model="embedding-2", input=sentence)
                embedding_list = response.data[0].embedding
            sentences_vec.append(embedding_list)  # 输出字典,'embedding':每个向量长度为1024的列表

        sentences[faiss_index_name] = sentences_vec

    else:  # sentences已经包含了向量化后的空间向量
        if isinstance(sentences, dict):
            faiss_index_name = embedded_vec_key
        elif isinstance(sentences, list) and all(
            isinstance(x, (int, float)) for x in sentences
        ):
            sentences = {faiss_index_name: sentences}
        elif isinstance(sentences, np.ndarray):  # 2维向量,还是一维向量,此处标记,以后可能要修改
            sentences = {faiss_index_name: sentences}

    # faiss检索:
    dataset = Dataset.from_dict(sentences)
    dataset.add_faiss_index(column=faiss_index_name)

    # 处理query,判断: str,或者一维ndarry, 或者 全是数字的列表;满足条件,query提出txt,向量化,非batch检索(单query检索)
    faiss_query = query
    if (
        isinstance(query, str)
        or (isinstance(query, np.ndarray) and np.ndim(query) == 1)
        or (isinstance(query, list) and all(isinstance(x, (int, float)) for x in query))
    ):
        # 当query 字符串时 向量化
        if isinstance(query, str):
            if query == "":  # 如果是空字符,则伪造一个1024维度的向量,取值范围为-1,1之间;
                embedding_list = np.random.uniform(-1, 1, size=(1024,)).tolist()
            else:
                response = client.embeddings.create(
                    model="embedding-2", input=query
                )  # 填写需要调用的模型名称
                embedding_list = response.data[0].embedding

            faiss_query = np.array(
                embedding_list, dtype=np.float32
            )  # get_nearest_examples(需要是numpy)

        if return_indices_samples:  # 返回检索位置
            scores, indices = dataset.search(
                index_name=faiss_index_name, query=faiss_query, k=k
            )
            return scores, indices
        else:  # 返回检索样本
            scores, nearest_examples = dataset.get_nearest_examples(
                index_name=faiss_index_name, query=faiss_query, k=k
            )
            return scores, nearest_examples

    # query,判断: List[str],或者2维向量,或者List[一维向量]
    elif (
        (isinstance(query, list) and all(isinstance(x, str) for x in query))
        or (isinstance(query, np.ndarray) and np.ndim(query) == 2)
        or (isinstance(query, list) and all(isinstance(x, (int, float)) for x in query))
    ):
        # 当query 为字符串列表时, 向量化
        if isinstance(query, list) and all(isinstance(x, str) for x in query):
            faiss_query = []
            for sentence in query:
                if sentence == "":  # 如果是空字符,则伪造一个1024维度的向量,取值范围为-1,1之间;
                    embedding_list = np.random.uniform(-1, 1, size=(1024,)).tolist()
                else:
                    response = client.embeddings.create(
                        model="embedding-2", input=sentence
                    )  # 填写需要调用的模型名称
                    embedding_list = response.data[0].embedding
                faiss_query.append(embedding_list)
            faiss_query = np.array(
                faiss_query, dtype=np.float32
            )  # get_nearest_examples(需要是numpy)

        if return_indices_samples:  # 返回检索位置
            scores, indices_batch = dataset.search_batch(
                index_name=faiss_index_name, queries=faiss_query, k=k
            )  # scores与indices_batch都是ndarry,shape=(n,1)
            return scores, indices_batch
        else:  # 返回检索样本
            scores, nearest_examples_batch = dataset.get_nearest_examples_batch(
                index_name=faiss_index_name, queries=faiss_query, k=k
            )
            return scores, nearest_examples_batch


def mark_semantic_index(
    df1: pd.DataFrame = None,
    df2: pd.DataFrame = None,
    university_code: int = 1,
    zhipuai_client: object = None,
    index_semantic_column: str = "index_NewQ_OldL",
    score_semantic_column: str = "score_NewQ_OldL",
):
    """
    以df1专业名称为Query,df2专业名称为List,寻找最接近的语义;在df1的Excel上,增加列"index_NewQ_OldL"与"score_OldQ_NewL",表明今年的专业名称在去年的专业名称中的对应的序列位置;当两个专业语义不通时,序列位置标记为-1;
    :param df1,df2: source and target DataFrame
    :param university_code:
    :param zhipuai_client:
    :param index_semantic_column,score_semantic_column:新增加了自定义列名;
    :return:
    """
    major1 = df1[df1["学校代号"] == university_code]["专业名称"]
    df2_university_list = df2["学校代号"].unique()
    if university_code in df2_university_list:
        major2 = df2[df2["学校代号"] == university_code]["专业名称"]
    else:
        return df1
    # replace NaN values with an empty string
    major1 = major1.fillna("")
    major2 = major2.fillna("")

    scores_list, indices_list = semantic_batch_search(
        zhipuai_client,
        query=major1.tolist(),
        sentences=major2.tolist(),
        k=1,
        return_indices_samples=True,
    )

    # nearest_list= [major2[indices[0]] for indices in indices_list]
    # print(f"Scores_list:{scores_list}")
    # print(f"indices_list:{indices_list}")
    # print(f"nearest_list:{nearest_list}")

    semantic_index = []
    semantic_score = []
    for i in range(len(scores_list)):
        score, index = scores_list[i][0], indices_list[i][0]
        if score > 0.84:  # 工科实验班(海洋)与海洋科学,空间距离为0.8369
            index = -1  # -1为特殊值,表示没有这个序列位置
        semantic_index.append(index)
        semantic_score.append(score)

    if index_semantic_column not in df1.columns:
        df1[index_semantic_column] = np.full(
            shape=(df1.shape[0],), fill_value=-1, dtype=int
        )
    if score_semantic_column not in df1.columns:
        df1[score_semantic_column] = np.full(
            shape=(df1.shape[0],), fill_value=99, dtype=float
        )
    df1.loc[df1["学校代号"] == university_code, index_semantic_column] = semantic_index
    df1.loc[df1["学校代号"] == university_code, score_semantic_column] = semantic_score

    return df1


def df2Excel_SemanticIndex(
    zhipuai_client,
    df1_excelPath,
        df1_sheetName,
    df2_excelPath,
        df2_sheetName,
    df1_outPath,
    df2_outPath,
    sheet_name="semantic",
):
    """
    读取两个Excel文件,df1,df2,分别代表source和target,通过zhipuai_client,对df1的"专业名称"列与df2的"专业名称"列进行语义比较,并Mark匹配index;然后分别写入到两个Excel文件中;
    1) 读取两个Excel成DataFrame;
    2) 语义比较,并Mark匹配index;
    3) 对每个df中每个学校代号,语义比较,并Mark匹配index;两张表,互相比较;
    4) 写入指定的excel文件,指定的sheet表中,
    :param zhipuai_client:
    :param df1_excelPath:
    :param df2_excelPath:
    :param df1_outPath:
    :param df2_outPath:
    :param sheet_name:
    :return: 写入excel文件,无返回;
    """
    path_list = [df1_excelPath, df2_excelPath ]
    # 检查文件是否存在，并且具有写权限
    for path in path_list:
        if not os.path.exists(path):
            print(f"{path}, doesn't exist.")
            return exit(1)

    # 检查xlsx文件是否已经有临时文件打开,从而判断是否EXCEL文件已经打开:
    path_list = [df1_outPath,df2_outPath]
    for path in path_list:
        if os.path.exists(path):
            if not os.access(path, os.W_OK):
                print(f"{path},无写权限.")
                return exit(1)
        dirname, filename = os.path.split(path)
        filename = "~$".join(["", filename])
        temp_path = os.path.join(dirname, filename)
        if os.path.exists(temp_path):
            print(f"{path},文件已经打开,禁止写入.")
            return exit(1)

    # 取Excel文件
    df1 = pd.read_excel(df1_excelPath,sheet_name=df1_sheetName)
    df2 = pd.read_excel(df2_excelPath,sheet_name=df2_sheetName)
    # 对每个df中每个学校代号,语义比较,并Mark匹配index;两张表,互相比较;
    df1_university_arr = df1["学校代号"].unique()
    df2_university_arr = df2["学校代号"].unique()
    university_arr = pd.unique(np.r_[df1_university_arr, df2_university_arr])

    start_time = time.time()
    for u in university_arr:
        if u in df1_university_arr:  # 排除不存在的univeristy导致query为[]
            df_query = mark_semantic_index(
                df1,
                df2,
                university_code=u,
                zhipuai_client=zhipuai_client,
                index_semantic_column=f"index_{df1_shortname}Q{df2_shortname}L",
                score_semantic_column=f"score_{df1_shortname}Q{df2_shortname}L",
            )
            print(f"{df1_shortname}->{df2_shortname}: university_code:{u}, completed.")
        if u in df2_university_arr:
            df_list = mark_semantic_index(
                df2,
                df1,
                university_code=u,
                zhipuai_client=zhipuai_client,
                index_semantic_column=f"index_{df2_shortname}Q{df1_shortname}L",
                score_semantic_column=f"score_{df2_shortname}Q{df1_shortname}L",
            )
            print(f"{df2_shortname}->{df1_shortname}: university_code:{u}, completed.")

    time_assumed = time.time() - start_time
    print(f"{df1_shortname}<->{df2_shortname} both spent {time_assumed/60:.2f}min ")

    # 判断文件新建,还是续写;写入指定的excel文件,指定的sheet表中,
    for path, df in zip([df1_outPath, df2_outPath],[df_query,df_list]):
        if os.path.exists(path):
            # mode='append',避免原有的sheet表中数据被清除;
            params = { 'path': path,
                        'mode': 'a',
                        'if_sheet_exists': 'replace',
                        'engine': 'openpyxl'}
        else:
            # 新建文件并写入
            params = { 'path': path,
                        'mode': 'w',
                        'engine': 'openpyxl'}

        with pd.ExcelWriter(**params) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, na_rep="")





def recover_major(df1_x:pd.Series, df1_index_semantic_column, df2:pd.DataFrame, recover_type:str='name'):
    """
    用于apply函数:
    将df1的每一行:df1_x,读出df1_index_semantic, 找到df2对应学校代号下的指定index的专业名称,再输出df2的专业名称,或者df2的专业代号;
    :param df1_x: pd.Series; df1的行
    :param df2: pd.DataFrame; df2
    :param index_semantic_column: df1对应的df1_index_semantic_column列名
    :param recover_type: str; "name" or "code",即,输出专业名称,还是专业代码
    :return: str; 专业名称,或者专业代号.
    """
    major_name_arr = df2[df2['学校代号'] == df1_x['学校代号']]['专业名称'].values #pd.Series转换成ndarry
    major_code_arr = df2[df2['学校代号'] == df1_x['学校代号']]['专业代号'].values #pd.Series转换成ndarry

    major = ""
    if major_name_arr.size != 0 or major_name_arr is not None:
        index = df1_x[df1_index_semantic_column]
        if index != -1 and index != np.NAN:
            if recover_type=='name':
                major = major_name_arr[index]
            elif recover_type=='code':
                major = major_code_arr[index]
    return major


if __name__ == "__main__":
    # 准备zhipuai client:
    config_path_zhipuai = r"L:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    # 定义文件名称: df1与df2的年份需要相邻,并且df1为今年/新年,而df2为去年/旧年;
    df1_shortname = "2024" # 今年/新年
    df2_shortname = "2023" # 去年/旧年
    # excel_path_base = "E:/Working Documents/装修/丁翊弘学习/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.{}"
    excel_path_base = "L:/丁翊弘/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.{}"
    df1_excelPath = excel_path_base.format(df1_shortname,'xls')
    df1_sheetName = "tdx2024"
    df2_excelPath = excel_path_base.format(df2_shortname,'xlsx')
    df2_sheetName = f"combine{df2_shortname}"
    df1_outPath = os.path.splitext(df1_excelPath)[0] + ".xlsx"
    df2_outPath = os.path.splitext(df2_excelPath)[0] + ".xlsx"

    # df1,df2,相互检索对方,对每个大学,匹配专业代号,写入excel文件:
    df2Excel_SemanticIndex(zhipuai_client=zhipuai_client,
                     df1_excelPath=df1_excelPath,
                    df1_sheetName=df1_sheetName,
                     df2_excelPath=df2_excelPath,
                    df2_sheetName=df2_sheetName,
                     df1_outPath=df1_outPath,
                     df2_outPath=df2_outPath,
                     sheet_name="semantic")

    # 发现:
    # 2022年与2021年语义比较:
    # 三步实现双向重复的semantic_index合并,以解决2022年某几个专业源自2021年拆解,以及2022年某个专业是2021年某几个专业合并而来
    # Step1: 2021做Query, 同学校有几个专业合并成2022年另个专业;解决: 对2021年,groupby('semantic_index),当semantic_index!=-1时,计划数sum,分数线mean,位次mean.;这样将2021表中相同semantic_index行合并;
    # Step2: 2022做Query, 同学校有几个专业自2021年拆解而来;解决: 1) 按照"学校代号""专业代号""semantic_index"三者条件,读取2021合并表后的对应的semantic_index对应的序列,获得二次合并表;
    # Step3: 2) 二次合并后的2022/2021表,再对同学校同专业同semantic_index进行groupby,然后对分组后的计划数加和,分数线mean,位次mean.

    # # Step 0: df1与df2相互进行query并生成index,score列; 即: a)去年做query, 生成index_OldQ_NewL, score_OldQ_NewL;b)今年做query,生成index_NewQ_OldL
    # # Step 1: 去年(旧年)做Query的index时, 今年(新年)有同学校的某一个专业为去年(旧年)合并而来;解决: groupby(semantic_index!=-1).agg()
    df1_index_semantic_column = f"index_{df1_shortname}Q{df2_shortname}L"
    df1_score_semantic_column = f"score_{df1_shortname}Q{df2_shortname}L"
    df1_sheetName = 'semantic'

    df2_index_semantic_column = f"index_{df2_shortname}Q{df1_shortname}L"
    df2_score_semantic_column = f"score_{df2_shortname}Q{df1_shortname}L"
    df2_sheetName = 'semantic'

    # sheet_name不指定,读第一个sheet;None:读所有的sheets,以字典输出;
    df1 = pd.read_excel(df1_outPath,sheet_name=df1_sheetName)
    df2 = pd.read_excel(df2_outPath, sheet_name=df2_sheetName)
    # 增加一列还原index_semantic对应的专业名称,便于观察语义匹配;
    df2[f'{df1_shortname}版专业名称_{df2_shortname}专业'] = df2.apply(recover_major,args=(df2_index_semantic_column,df1,'name'),axis=1)
    df1[f'{df2_shortname}版专业名称_{df1_shortname}专业'] = df1.apply(recover_major,args=(df1_index_semantic_column,df2,'name'),axis=1)
    # 由于每年的专业代号在增减专业后,会重新编号,所以需要以新年(今年)的专业代号为基准,统一今年/旧年的专业代号,才可以合并,今年与旧年的表格
    df2[f'{df1_shortname}版专业代号_{df2_shortname}专业'] = df2.apply(recover_major, args=(df2_index_semantic_column, df1, 'code'), axis=1)
    df1[f'{df2_shortname}版专业代号_{df1_shortname}专业'] = df1.apply(recover_major, args=(df1_index_semantic_column, df2, 'code'), axis=1)

    df2[df2_index_semantic_column] = df2[df2_index_semantic_column].apply(
        lambda x: np.nan if x == -1 else x
    ) # -1转化为Nan,因为agg()不对Nan进行计算
    grouped_df21 = df2.groupby(
        by=["学校代号", df2_index_semantic_column], sort=False, dropna=False) # dropna=False 保留Nan行
    # 使用字典对每列agg()函数;未定义的列缺省是first()
    aggFunc_dict = {}
    columns = list(set(df2.columns)^{'学校代号',df2_index_semantic_column}) # 减除groupby的作为index的条件列
    for c in columns:
        if "名称" in c: # 包括学校名称,专业名称
            aggFunc_dict[c] = lambda x: x
        elif "计划数" in c:
            aggFunc_dict[c] = lambda x: np.nan if pd.isna(x.sum()) else int(x.sum())
        elif "分数线" in c or "位次" in c:
            aggFunc_dict[c] = lambda x: np.nan if pd.isna(x.mean()) else int(x.mean())
        else:
            aggFunc_dict[c] = "first"

    combine_oldyear = grouped_df21.agg(aggFunc_dict).reset_index()
    # #或使用,pd.NamedAgg: 优点,可以直接定义更改后的列名
    # combine_ = grouped.agg(**{'平均分数线':('分数线', 'mean'),
    #                          '平均位次':('位次','mean')})
    # 列排序:
    focus_columns = [
        "学校代号",
        "学校名称",
        "专业代号",
        f'{df1_shortname}版专业代号_{df2_shortname}专业',
        "专业名称",
        f'{df1_shortname}版专业名称_{df2_shortname}专业',
        "计划数",
        "分数线",
        "位次",
        df2_index_semantic_column,
        df2_score_semantic_column,
    ]

    focus_columns.extend([c for c in df2.columns if c not in focus_columns and c !='_merge']) # 同时移除"_merge"列,否则再次indicator=True会报错(can't use name of existing column for indicator column)

    combine_oldyear = combine_oldyear[focus_columns[:]]

    # pd.to_excel(df2_outPath, )会导致excel原有的sheet的数据都丢失;需要通过pd.ExcelWriter(mode='append')
    if os.path.exists(df2_outPath):
        params = {'path': df2_outPath, 'mode': 'a', 'if_sheet_exists': 'replace', 'engine': "openpyxl"}
    else:
        params = {'path': df2_outPath, 'mode': 'w', 'engine': "openpyxl"}
    with pd.ExcelWriter(**params) as writer:
        combine_oldyear.to_excel(writer, sheet_name=f'temp{df2_shortname}', index=False, na_rep="")

    # # Step2: 今年(新年)做Query,去年(旧年)做List,今年有几个专业是由去年某个专业拆解而来;解决: pd.merge()
    # 检查是否有_merge列,有则移除该列, 否则再次indicator=True会报错(can't use name of existing column for indicator column)
    combine_newyear = pd.merge(df1, combine_oldyear, left_on=['学校代号','专业代号'], right_on=['学校代号',f'{df1_shortname}版专业代号_{df2_shortname}专业'],
                        how='outer', indicator=True, suffixes=(f'_{df1_shortname}',f'_{df2_shortname}'),validate='one_to_many')
    # 更改关键列列名: 关键列名去除今年的suffix,用于下一轮循环;
    combine_newyear = combine_newyear.rename(columns={f"学校名称_{df1_shortname}":'学校名称',
                                    f"专业代号_{df1_shortname}":'专业代号',
                                    f"专业名称_{df1_shortname}":'专业名称',
                                    f"计划数_{df1_shortname}":'计划数',
                                    f"分数线_{df1_shortname}":'分数线',
                                    f"位次_{df1_shortname}":'位次'}) # 列名去除今年的suffix,用于下一轮循环;
    # '_merge'列中标明了"left_only",'both',"right_only";去除'right_only'行:
    combine_newyear = combine_newyear[combine_newyear['_merge'] != 'right_only']
    focus_columns = [
                "学校代号",
                "学校名称",
                "专业代号",
                "专业名称",
                "计划数",
                f"计划数_{df2_shortname}",
                "分数线",
                f"分数线_{df2_shortname}",
                "位次",
                f"位次_{df2_shortname}",
    ]
    diff_columns = list(set(combine_newyear.columns)^set(focus_columns) ^ {df1_score_semantic_column,
                                                                      df1_index_semantic_column,
                                                                      df2_index_semantic_column,
                                                                      df2_score_semantic_column}) # 移除不必要显示的列
    focus_columns.extend(diff_columns)
    combine_newyear = combine_newyear[focus_columns[:]]
    sheet_name = f'combine{df1_shortname}'
    if os.path.exists(df1_outPath):
        params = {'path': df1_outPath, 'mode': 'a', 'if_sheet_exists': 'replace', 'engine': "openpyxl"}
    else:
        params = {'path': df1_outPath, 'mode': 'w', 'engine': "openpyxl"}
    with pd.ExcelWriter(**params) as writer:
        combine_newyear.to_excel(writer, sheet_name=sheet_name, index=False, na_rep="")

    print(f'3 steps:\n1)向量模型语义匹配\n2){df2_outPath}.temp{df2_shortname}写入\n;3){df1_outPath}写入{sheet_name}\n完成!')

    ## ---------------------------------------------------------------
    # # 更改列序,重新生成combine2024_sort
    path = excel_path_base.format('高考统计分析2018_2024', 'xlsx')
    columns_sort = ['学校代号', '学校名称', '专业代号', '专业名称',
                    '分数线', '分数线_2023','分数线_2022', '分数线_2021', '分数线_2020', '分数线_2019', '分数线_2018',
                    '位次', '位次_2023','位次_2022', '位次_2021', '位次_2020', '位次_2019', '位次_2018',
                    '计划数', '计划数_2023','计划数_2022', '计划数_2021', '计划数_2020', '计划数_2019', '计划数_2018',
                    '2024版专业名称_2023专业', '2023版专业名称_2024专业',
                    '2023版专业名称_2022专业', '2022版专业名称_2023专业',
                    '2022版专业名称_2021专业', '2021版专业名称_2022专业',
                    '2021版专业名称_2020专业', '2020版专业名称_2021专业',
                    '2020版专业名称_2019专业', '2019版专业名称_2020专业',
                    '2019版专业名称_2018专业', '2018版专业名称_2019专业',
                    '学校名称_2022', '学校名称_2021', '学校名称_2020', '学校名称_2019', '学校名称_2018']
    df = df1_outPath[columns_sort[:]]
    sheet_name_sort = 'combine2023_sort'
    if os.path.exists(path):
        params = {'path': path, 'mode': 'a', 'if_sheet_exists': 'replace', 'engine': "openpyxl"}
    else:
        params = {'path': path, 'mode': 'w', 'engine': "openpyxl"}
    with pd.ExcelWriter(**params) as writer:
        df.to_excel(writer, sheet_name=sheet_name_sort, index=False, na_rep="")
    # # ----------------------------------------------------------------

1
