import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from typing import Union,List
from mysql_handler import config_read


def semantic_search(
    client,
    query: Union[str,List[str], np.ndarray],
    sentences: Union[list, dict, pd.Series],
    embedding_txt_key: str = None,
    embedded_vec_key: str = None,
    k: int = 3,
        use_glm_Embedding2:bool= True,
    return_indices_samples: bool = False,
):
    """
    根据query,以及sentences是否是已经向量化后的向量空间,来判断faiss方法中是使用文本搜索,还是向量搜索;query为向量时,sentences应也包括同等向量算法向量化后的向量空间;当query为文本时,可以根据use_glm_embedding2参数,选择使用文本搜索,还是使用glm的向量模型Embedding-2进行向量化后,再进行向量空间搜索.
    query可以是单一字符串或者单一向量,在sentences列中搜索与query最相似的k个句子; 也可以是一个字符串列表,或者向量矩阵,搜索语义最接近的k个句子.(使用huggingface的Dataset调用的faiss方法);
    1) query: Union[str,List[str], np.ndarray]; 可以是单个字符串,也可以是字符串列表,也可以直接是向量化后的向量,或者向量矩阵
    1) sentences为一字典时,则至少包含待embedding的文本key=embedding_txt_key,根据use_glm_Embedding2来判断是否转换向量空间,然后字典再转换成huggingface的Dataset;
       sentences字典,如果已经包含了embedded之后的向量,即包含向量化后的key,(key=embedded_vec_key),则无需经过向量化过程,直接进入向量空间语义搜索;
    2) sentences为list时,则列表内包含的或是待embedding的字符串,则转换成字典,key='embedding_txt';或是一维向量,或者二维向量,则转化成字典,key='embedding';
    3) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本;或者serach()获得最佳的k个dataset的样本的位置
    :param:
        client: object; zhipuAI client (已经送入API_KEY)
        query: str; 欲搜索的关键词或者句子
        sentences: 列表或者字典,送入embedding-2模型,获得长度1024的向量列表;key:列名;value:文本或者向量;setences的key如果有embedded_vec_key,其值必须是已经转化后的向量空间
        embedding_txt_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索;
        embedded_vec_key: str; sentences字典中包含的已经向量化后的向量空间列的key;
        k: int; 返回最相似的句子数量
        use_glm_Embedding2: bool;是否使用glm的向量模型Embedding-2进行向量化;True: 使用Embedding-2进行向量化;False: 使用文本搜索
        return_indices_samples: bool;是否返回检索后在序列中的位置;True: 返回检索位置;False:返回nearest_samples
    :return:
        scores, nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳)
    """
    sentences_txt = []
    sentences_vec = []


    # 需要判断非二维ndarry,非list;即,或str,或一维ndarray: 单一维ndarry时,sentences也必须包含向量化后的向量空间,即setences[embedded_vec_key],无需向量化了,需改动这里,
    if isinstance(query,str): # 单一字符串,非向量,做single检索
        faiss_query = query
        if use_glm_Embedding2:
            faiss_index_name = "_embedding"
            # sentences 字典提出sentences_txt
            if isinstance(sentences, dict):
                if embedding_txt_key is None: # 未给待embedding_txt列的列名,取第一列
                    embedding_txt_key = list[sentences.keys()][0]
                    sentences_txt = sentences[embedding_txt_key]
                else:
                    sentences_txt = sentences[embedding_txt_key]

            elif isinstance(sentences, list):
                sentences_txt = sentences
                sentences = {"_embedding_txt": sentences_txt}
            elif isinstance(sentences, pd.Series):
                sentences_txt = sentences.values
                sentences = {"_embedding_txt": sentences_txt}

            # sentences 向量化
            # if embedded_vec_key is None: # sentences字典已经包含了向量化过的向量空间,则无需再向量化
            for sentence in sentences_txt:
                response = client.embeddings.create(model="embedding-2", input=sentence)
                sentences_vec.append(
                    response.data[0].embedding
                )  # 输出字典,'embedding':每个向量长度为1024的列表

            sentences[faiss_index_name] = sentences_vec
            # query 向量化
            response = client.embeddings.create(model="embedding-2", input=query)  # 填写需要调用的模型名称
            faiss_query = np.array(
                response.data[0].embedding, dtype=np.float32
            )  # get_nearest_examples(需要是numpy)
        else: # 文本检索,faiss非向量检索

            if embedding_txt_key is None: # 未给待embedding_txt列的列名,取第一列
                faiss_index_name = list[sentences.keys()][0]
            else:
                faiss_index_name = embedding_txt_key

            if isinstance(sentences, list):
                sentences = {"_embedding_txt": sentences}
            elif isinstance(sentences, pd.Series):
                sentences_txt = sentences.values
                sentences = {"_embedding_txt": sentences_txt}

        # faiss检索:
        dataset = Dataset.from_dict(sentences)
        dataset.add_faiss_index(column="embedding")

        if return_indices_samples:  # 返回检索位置
            scores, indices = dataset.search(index_name=faiss_index_name, query=faiss_query, k=k)
            return scores, indices
        else:  # 返回检索样本
            scores, nearest_examples = dataset.get_nearest_examples(findex_name=faiss_index_name, query=faiss_query, k=k )
            return scores, nearest_examples





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

    for major in major1:
        scores, indices = semantic_search(
            zhipuai_client,
            query=major,
            sentences=major2,
            k=1,
            return_indices_samples=True,
        )
        scores_list.append(scores[0])
        indices_list.append(indices[0])
        nearest_list.append(major2[indices[0]])

    result = {
        "source_txt": major1,
        "target_txt": major2,
        "scores_list": scores_list,
        "nearest_list": nearest_list,
    }

    print(f"Scores_list:{scores_list}")
    print(f"indices_list:{indices_list}")
    print(f"nearest_list:{nearest_list}")
    # print(result)
