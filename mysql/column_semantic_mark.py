import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from typing import Union, List
from mysql_handler import config_read
import os


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
    :param:
        client: object; zhipuAI client (已经送入API_KEY)
        query: Union[str,List[str], np.ndarray]; 欲搜索的关键词或者句子,或者向量,或者字符串列表;query不应有空字符串出现
        sentences: 列表或者字典,送入embedding-2模型,获得长度1024的向量列表;key:列名;value:文本;不应有空字符串出现
        embedding_txt_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索;当sentences为list,可为None;sentence为字典时,缺省第一列可设为None
        embedded_vector_key: str: 如果预存有向量化后的向量空间,此处为其key名;
        k: int; 返回最相似的句子数量
        return_indices_samples: bool;是否返回检索后在序列中的位置;True: 返回检索位置;False:返回nearest_samples
    :return:
        scoresList[List[float], nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳),类型分别为: List[List[float], List[dict],
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


if __name__ == "__main__":
    # 准备zhipuai client:
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    # 定义文件名称:
    df1_shortname = "2022"
    df2_shortname = "2021"
    excel_path_base = "E:/Working Documents/装修/丁翊弘学习/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.xls"
    # excel_path_base = "L:/丁翊弘/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.xls"
    df1_excelPath = excel_path_base.format(df1_shortname)
    df2_excelPath = excel_path_base.format(df2_shortname)
    df1_outPath = os.path.splitext(df1_excelPath)[0] + ".xlsx"
    df2_outPath = os.path.splitext(df2_excelPath)[0] + '.xlsx'
    # # 读取Excel文件
    # df1 = pd.read_excel(df1_excelPath)
    # df2 = pd.read_excel(df2_excelPath)
    # # 语义比较,并Mark匹配index
    # university_list = df1["学校代号"].unique()
    #
    # for u in university_list:
    #     df_query = mark_semantic_index(
    #         df1,
    #         df2,
    #         university_code=u,
    #         zhipuai_client=zhipuai_client,
    #         index_semantic_column=f"index_{df1_shortname}Q{df2_shortname}L",
    #         score_semantic_column=f"score_{df1_shortname}Q{df2_shortname}L",
    #     )
    #     print(f"{df1_shortname}->{df2_shortname}: university_code:{u}, completed.")
    #     df_list = mark_semantic_index(
    #         df2,
    #         df1,
    #         university_code=u,
    #         zhipuai_client=zhipuai_client,
    #         index_semantic_column=f"index_{df2_shortname}Q{df1_shortname}L",
    #         score_semantic_column=f"score_{df2_shortname}Q{df1_shortname}L",
    #     )
    #     print(f"{df2_shortname}->{df1_shortname}: university_code:{u}, completed.")
    #
    # sheetName = "semantic"
    # df_query.to_excel(df1_outPath, sheet_name=sheetName, index=False, na_rep="")
    # df_list.to_excel(df2_outPath, sheet_name=sheetName, index=False, na_rep="")

    # 发现:
    # 2022年与2021年语义比较:
    # 三步实现双向重复的semantic_index合并,以解决2022年某几个专业源自2021年拆解,以及2022年某个专业是2021年某几个专业合并而来
    # Step1: 2021做Query, 同学校有几个专业合并成2022年另个专业;解决: 对2021年,groupby('semantic_index),当semantic_index!=-1时,计划数sum,分数线mean,位次mean.;这样将2021表中相同semantic_index行合并;
    # Step2: 2022做Query, 同学校有几个专业自2021年拆解而来;解决: 1) 按照"学校代号""专业代号""semantic_index"三者条件,读取2021合并表后的对应的semantic_index对应的序列,获得二次合并表;
    # Step3: 2) 二次合并后的2022/2021表,再对同学校同专业同semantic_index进行groupby,然后对分组后的计划数加和,分数线mean,位次mean.

    # Step 0: df1与df2相互进行query并生成index,score列; 即: a)去年做query, 生成index_OldQ_NewL, score_OldQ_NewL;b)今年做query,生成index_NewQ_OldL
    # Step 1: 去年(旧年)做Query的index时, 今年(新年)有同学校的某一个专业为去年(旧年)合并而来;解决: groupby(semantic_index!=-1).sum()
    index_semantic_column = f"index_{df1_shortname}Q{df2_shortname}L"
    combine = pd.read_excel(df1_outPath, nrows=100)

    combine[index_semantic_column] = combine[index_semantic_column].apply(lambda x: np.nan if x==-1 else x)
    grouped =combine.groupby(by=["学校代号","专业代号",index_semantic_column],sort=False,dropna=True)
    result = grouped.agg()

