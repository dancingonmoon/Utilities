import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from typing import Union
from mysql_handler import config_read


def semantic_search(
    client,
    query: str,
    sentences: Union[list, dict, pd.Series],
    embedding_txt_key: str = None,
    embedded_vec_key: str = None,
    k: int = 3,
    return_indices_samples: bool = False,
):
    """
    使用zhipuAI向量模型Embedding-2, 在sentences列中搜索与query最相似的k个句子.(使用huggingface的Dataset调用的faiss方法)
    1) sentences为一字典时,则至少包含待embedding的文本key=embedding_txt_key,将其转换成huggingface的Dataset;
       sentences字典,如果已经包含了已经embedded之后的向量,即已经向量化后的key,(key=embedded_vec_key),则无需经过向量化过程,直接进入语义搜索;
    2) sentences为list时,则列表内包含的就是待embedding的txt;转换成字典,key='embedding_txt';
    3) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本
    :param:
        client: object; zhipuAI client (已经送入API_KEY)
        query: str; 欲搜索的关键词或者句子
        sentences: 列表或者字典,送入embedding-2模型,获得长度1024的向量列表;key:列名;value:文本或者向量;setences的key如果有embedded_vec_key,其值必须是已经转化后的向量空间
        embedding_txt_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索;
        embedded_vec_key: str; sentences字典中包含的已经向量化后的向量空间列的key;
        k: int; 返回最相似的句子数量
        return_indices_samples: bool;是否返回检索后在序列中的位置;True: 返回检索位置;False:返回nearest_samples
    :return:
        scores, nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳)
    """
    sentences_vec = []
    sentences_txt = []
    # 如果sentences列表,取列表待embedding文本后,转换成字典,key随便取'embedding_txt';
    # 如果sentences是字典,如果没有embedding_key则,假设第一列为待embedding的文本;
    # 如果sentences是字典,已经embedding获得了向量空间,列名是embedded_vec_key,则跳过空间向量生成过程,直接语义搜索
    if isinstance(sentences, dict):
        if not embedded_vec_key in sentences:
            if embedding_txt_key is not None:
                sentences_txt = sentences[embedding_txt_key]
            else:  # 未给待embedding列的列名,取第一列
                sentences_txt = sentences[list(sentences.keys())[0]]
        else:  # 已经包含了向量化的列,key=embedded_vec_key
            sentences["embedding"] = sentences[embedded_vec_key]
    elif isinstance(sentences, list):
        sentences_txt = sentences
        sentences = {"embedding_txt": sentences_txt}
    elif isinstance(sentences, pd.Series):
        sentences_txt = sentences.values

    for sentence in sentences_txt:
        response = client.embeddings.create(model="embedding-2", input=sentence)
        sentences_vec.append(
            response.data[0].embedding
        )  # 输出字典,'embedding':每个向量长度为1024的列表

    sentences["embedding"] = sentences_vec
    dataset = Dataset.from_dict(sentences)
    dataset.add_faiss_index(column="embedding")

    response = client.embeddings.create(model="embedding-2", input=query)  # 填写需要调用的模型名称
    query_embedded_vec = np.array(
        response.data[0].embedding, dtype=np.float32
    )  # get_nearest_examples(需要是numpy)

    if return_indices_samples:  # 返回检索位置
        scores, indices = dataset.search("embedding", query_embedded_vec, k=k)
        return scores, indices
    else:  # 返回检索样本
        scores, nearest_examples = dataset.get_nearest_examples(
            "embedding", query_embedded_vec, k=k
        )
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
