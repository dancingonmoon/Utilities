import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from typing import Union, List
from mysql_handler import config_read


def semantic_search(
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
    1) query: Union[str,List[str], np.ndarray]; 可以是单个字符串,也可以是字符串列表,也可以直接是向量化后的向量,或者向量矩阵
    1) sentences为一字典时,则包含待embedding的文本key=embedding_txt_key,然后字典再转换成huggingface的Dataset;也可以包括向量化后的向量空间,以key=embedded_vector_key表示;
    2) sentences为list时,则列表内包含的或是待embedding的字符串,则转换成字典,key='_embedding_txt';;
    3) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本;或者serach()获得最佳的k个dataset的样本的位置
    :param:
        client: object; zhipuAI client (已经送入API_KEY)
        query: Union[str,List[str], np.ndarray]; 欲搜索的关键词或者句子,或者向量,或者字符串列表;
        sentences: 列表或者字典,送入embedding-2模型,获得长度1024的向量列表;key:列名;value:文本;
        embedding_txt_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索;
        embedded_vector_key: str: 如果预存有向量化后的向量空间,此处为其key名;
        k: int; 返回最相似的句子数量
        return_indices_samples: bool;是否返回检索后在序列中的位置;True: 返回检索位置;False:返回nearest_samples
    :return:
        scores, nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳)
        或者 scores, indices
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
    if embedded_vec_key is None and faiss_index_name not in sentences:  # sentences字典已经包含了向量化过的向量空间,则无需再向量化
        for sentence in sentences_txt:
            response = client.embeddings.create(model="embedding-2", input=sentence)
            sentences_vec.append(
                response.data[0].embedding
            )  # 输出字典,'embedding':每个向量长度为1024的列表

        sentences[faiss_index_name] = sentences_vec

    else:  # sentences已经包含了向量化后的空间向量
        if isinstance(sentences, dict):
            faiss_index_name = embedded_vec_key
        elif isinstance(sentences, list) and all(isinstance(x, (int, float)) for x in sentences):
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
            response = client.embeddings.create(
                model="embedding-2", input=query
            )  # 填写需要调用的模型名称
            faiss_query = np.array(
                response.data[0].embedding, dtype=np.float32
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
                response = client.embeddings.create(
                    model="embedding-2", input=query
                )  # 填写需要调用的模型名称
                faiss_query.append(response.data[0].embedding)
            faiss_query = np.array(
                faiss_query, dtype=np.float32
            )  # get_nearest_examples(需要是numpy)

        if return_indices_samples:  # 返回检索位置
            scores, indices_batch = dataset.search_batch(
                index_name=faiss_index_name, queries=faiss_query, k=k ) # 运行有误, 得出都为0 ,需检查batch用法;
            return scores, indices_batch
        else:  # 返回检索样本
            scores, nearest_examples_batch = dataset.get_nearest_examples_batch(
                index_name=faiss_index_name, queries=faiss_query, k=k )
            return scores, nearest_examples_batch


if __name__ == "__main__":
    # 准备zhipuai client:
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)

    # 读取Excel文件
    # excel_path = r"l:\丁翊弘\高考\浙江省2021年普通高校招生普通类第一段平行投档分数线表.xls"
    excel_path1 = r"E:\Working Documents\装修\丁翊弘学习\高考\浙江省2021年普通高校招生普通类第一段平行投档分数线表.xls"
    excel_path2 = r"E:\Working Documents\装修\丁翊弘学习\高考\浙江省2022年普通高校招生普通类第一段平行投档分数线表.xls"

    df1 = pd.read_excel(
        excel_path1,
    )
    df2 = pd.read_excel(
        excel_path2,
    )

    university_code = 1  # 1 浙江大学
    major1 = df1[df1["学校代号"] == university_code]["专业名称"]
    major2 = df2[df2["学校代号"] == university_code]["专业名称"]
    # replace NaN values with an empty string
    major1 = major1.fillna("").tolist()[:5]
    major2 = major2.fillna("").tolist()

    scores_list = []
    indices_list = []
    nearest_list = []

    # for major in major1:
    #     scores, indices = semantic_search(
    #         zhipuai_client,
    #         query=major,
    #         sentences=major2,
    #         k=1,
    #         return_indices_samples=True,
    #     )
    #     scores_list.append(scores[0])
    #     indices_list.append(indices[0])
    #     nearest_list.append(major2[indices[0]])
    scores, indices = semantic_search(zhipuai_client,query=major1,sentences=major2, k=1, return_indices_samples=True)
    print(scores)
    print(indices)
    # result = {
    #     "source_txt": major1,
    #     "target_txt": major2,
    #     "scores_list": scores_list,
    #     "nearest_list": nearest_list,
    # }

    # print(f"Scores_list:{scores_list}")
    # print(f"indices_list:{indices_list}")
    # print(f"nearest_list:{nearest_list}")
    # print(result)
