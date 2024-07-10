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
    df2_excelPath,
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
    :return: 写入excel文件,无返回;
    """
    path_list = [df1_excelPath, df2_excelPath , df1_outPath,    df2_outPath,]
    # 检查文件是否存在，并且具有写权限
    for path in path_list:
        if not os.path.exists(path):
            print(f"{path},文件不存在")
            return
        if not os.access(path, os.W_OK):
            print(f"{path},无写权限.")
            return
    # 检查xlsx文件是否已经有临时文件打开,从而判断是否EXCEL文件已经打开:
    path_list = [df1_outPath,df2_outPath]
    for path in path_list:
        dirname, filename = os.path.split(path)
        filename = "~$".join(["", filename])
        temp_path = os.path.join(dirname, filename)
        if os.path.exists(temp_path):
            print(f"{path},文件已经打开,禁止写入.")
            return

    # 取Excel文件
    df1 = pd.read_excel(df1_excelPath)
    df2 = pd.read_excel(df2_excelPath)
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
    # sheetName = "semantic"
    df_query.to_excel(df1_outPath, sheet_name=sheet_name, index=False, na_rep="")
    df_list.to_excel(df2_outPath, sheet_name=sheet_name, index=False, na_rep="")


def recover_major(df1_x:pd.Series, df1_index_semantic_column, df2:pd.DataFrame, ):
    """
    用于apply函数:
    将df1的每一行:df1_x,读出df1_index_semantic, 找到df2对应学校代号下的指定index的专业名称,再输出
    :param df1_x: pd.Series; df1的行
    :param df2: pd.DataFrame; df2
    :param index_semantic_column: df1对应的df1_index_semantic_column列名
    :return: str; 专业名称
    """
    major_arr = df2[df2['学校代号'] == df1_x['学校代号']]['专业名称'].values #pd.Series转换成ndarry
    if major_arr.size == 0 or major_arr is None:
        major = ""
    else:
        index = df1_x[df1_index_semantic_column]
        if index != -1 and index != np.NAN:
            major = major_arr[index]
        else:
            major = ''
    return major


if __name__ == "__main__":
    # 准备zhipuai client:
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    # 定义文件名称:
    df1_shortname = "2019"
    df2_shortname = "2018"
    excel_path_base = "E:/Working Documents/装修/丁翊弘学习/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.xls"
    # excel_path_base = "L:/丁翊弘/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.xls"
    df1_excelPath = excel_path_base.format(df1_shortname)
    df2_excelPath = excel_path_base.format(df2_shortname)
    df1_outPath = os.path.splitext(df1_excelPath)[0] + ".xlsx"
    df2_outPath = os.path.splitext(df2_excelPath)[0] + ".xlsx"

    # # df1,df2,相互检索对方,对每个大学,匹配专业代号,写入excel文件:
    # df2mark_Excelout(zhipuai_client=zhipuai_client,
    #                  df1_excelPath=df1_excelPath,
    #                  df2_excelPath=df2_excelPath,
    #                  df1_outPath=df1_outPath,
    #                  df2_outPath=df2_outPath,
    #                  sheet_name="semantic"
    #                  )

    # 发现:
    # 2022年与2021年语义比较:
    # 三步实现双向重复的semantic_index合并,以解决2022年某几个专业源自2021年拆解,以及2022年某个专业是2021年某几个专业合并而来
    # Step1: 2021做Query, 同学校有几个专业合并成2022年另个专业;解决: 对2021年,groupby('semantic_index),当semantic_index!=-1时,计划数sum,分数线mean,位次mean.;这样将2021表中相同semantic_index行合并;
    # Step2: 2022做Query, 同学校有几个专业自2021年拆解而来;解决: 1) 按照"学校代号""专业代号""semantic_index"三者条件,读取2021合并表后的对应的semantic_index对应的序列,获得二次合并表;
    # Step3: 2) 二次合并后的2022/2021表,再对同学校同专业同semantic_index进行groupby,然后对分组后的计划数加和,分数线mean,位次mean.

    # # Step 0: df1与df2相互进行query并生成index,score列; 即: a)去年做query, 生成index_OldQ_NewL, score_OldQ_NewL;b)今年做query,生成index_NewQ_OldL
    # # Step 1: 去年(旧年)做Query的index时, 今年(新年)有同学校的某一个专业为去年(旧年)合并而来;解决: groupby(semantic_index!=-1).agg()
    index_semantic_column = f"index_{df2_shortname}Q{df1_shortname}L"
    score_semantic_column = f"score_{df2_shortname}Q{df1_shortname}L"

    df1 = pd.read_excel(df1_outPath)
    df2 = pd.read_excel(
        df2_outPath,
    )
    # 增加一列还原index_semantic对应的专业名称

    df2[f'{df1_shortname}专业名称'] = df2.apply(recover_major,args=(df1,),axis=1)
    df1[f'{df2_shortname}专业名称'] = df1.apply(recover_major,args=(df2,),axis=1)

    # print(df1, df2)

    df2[index_semantic_column] = df2[index_semantic_column].apply(
        lambda x: np.nan if x == -1 else x
    ) # -1转化为Nan,因为agg()不对Nan进行计算
    grouped = df2.groupby(
        by=["学校代号", index_semantic_column], sort=False, dropna=False
    ) # dropna=False 保留Nan行
    # 使用字典对每列定义函数:
    result = grouped.agg(
        {
            "学校名称": lambda x: x,
            "专业名称": lambda x: x,
            "专业代号": lambda x: x,
            "计划数": lambda x: int(x.sum()),
            "分数线": lambda x: int(x.mean()),
            "位次": lambda x: int(x.mean()),
            score_semantic_column: "min",
        }
    ).reset_index()
    # #或使用,pd.NamedAgg: 优点,可以直接定义更改后的列名
    # result_ = grouped.agg(**{'平均分数线':('分数线', 'mean'),
    #                          '平均位次':('位次','mean')})
    # 列排序:
    columns = [
        "学校代号",
        "学校名称",
        "专业代号",
        "专业名称",
        "计划数",
        "分数线",
        "位次",
        index_semantic_column,
        score_semantic_column,
    ]
    result = result[columns[:]]
    sheet_name = 'combine'
    result.to_excel(df2_outPath, sheet_name=sheet_name, index=False, na_rep="")
    #
    # # Step2: 今年(新年)做Query,去年(旧年)做List,今年有几个专业是由去年某个专业拆解而来;解决: pd.concat()
    # pd.merge(df1, df2, left_on=['学校代号',index_semantic_column], right_on=['学校代号',index_semantic_column], how='outer')
