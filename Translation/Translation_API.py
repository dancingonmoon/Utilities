import requests
import random
from hashlib import md5
import uuid

import re
import os
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


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


def MStranslation_API(
        text,
        lang_in: str = None,
        lang_out="zh-Hans",
        subscription_key="pls input your MStranslation API key",
):
    """
    指定文本的微软翻译:
    args:
        text: 待翻译的原文本,或者列表;
        lang_in : 翻译前语言; 当None或者不包含时,自动辨识语言
        lang_out: 翻译后语言;可以列表形式,增加多种语言,例如:['ru','zh-Hans']
        subscription_key: 微软翻译API的key;
    out:
        trans_text: 即翻译后的文本字典列表,列表中每个字典包含了单个语种,或者多个语种的'text':'trans_text'的键值对;
        response: API接口输出,当翻译出错时,观察错误信息;
    """

    # Add your subscription key and endpoint
    subscription_key = subscription_key
    endpoint = "https://api.cognitive.microsofttranslator.com"  # 认知服务
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = "eastasia"
    path = "/translate"
    constructed_url = endpoint + path

    params = {"api-version": "3.0", "from": lang_in, "to": lang_out}

    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    # You can pass more than one object in body.
    body = []

    if isinstance(text, list):
        for txt in text:
            body.append({"text": txt})

    if isinstance(text, str):
        body.append({"text": text})

    # print(body)
    # 当lang_out只有1种语言时,将lang_out变成列表:
    if isinstance(lang_out, str):
        lang_out = [lang_out]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    # response_json = json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
    # print(response_json)

    trans_text = []

    try:
        if isinstance(lang_out, list) and len(lang_out) > 1:
            trans_text = {}
            for i, lang in enumerate(lang_out):
                tmp = []
                for r in response:
                    tmp.append(r["translations"][i]["text"])
                trans_text[lang] = tmp
        else:
            for r in response:
                trans_text.append(r["translations"][0]["text"])
    except:
        print(response)

    return trans_text, response


def MStranslation_dynamicDictionary_API(
        text,
        dynamic_dict: dict = False,
        lang_in: str = None,
        lang_out: str = "zh-Hans",
        subscription_key="pls input your MStranslation API key",
):
    """
    指定文本的微软翻译:
    args:
        text: 待翻译的原文本,或者列表;
        dynamic_dict: 动态词典,包含有专有词汇,产品名称,人物人名等,已经有固定翻译的词汇,例:{'莫言':'Mr.Moyan'};缺省值:False,表示不需要动态词典;
        lang_in : 翻译前语言;当None或者不包含时,自动辨识语言
        lang_out: 翻译后语言;可以列表形式,增加多种语言,例如:['ru','zh-Hans']
        subscription_key: 微软翻译API的key;
    out:
        trans_text: 即翻译后的文本字典列表,列表中每个字典包含了单个语种,或者多个语种的'text':'trans_text'的键值对;
        response: API接口输出,当翻译出错时,观察错误信息;
    """

    # Add your subscription key and endpoint
    subscription_key = subscription_key
    endpoint = "https://api.cognitive.microsofttranslator.com"  # 认知服务
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = "eastasia"
    path = "/translate"
    constructed_url = endpoint + path

    params = {"api-version": "3.0", "from": lang_in, "to": lang_out}

    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    # You can pass more than one object in body.
    body = []

    if isinstance(text, list):
        for txt in text:
            if isinstance(dynamic_dict, dict):
                for key in dynamic_dict.keys():
                    # 如果txt中含有动态词典的每个key的词汇,需将其替换成对应的value
                    sub_txt = (
                            '<mstrans:dictionary translation="'
                            + dynamic_dict[key]
                            + '">'
                            + key
                            + "</mstrans:dictionary>"
                    )
                    txt = re.sub(key, sub_txt, txt)
                body.append({"text": txt})
            elif dynamic_dict == False or dynamic_dict == "":
                body.append({"text": txt})
            else:
                print("Neither False nor Dictionary dynamic_dict is !")

    if isinstance(text, str):  # text为一条文本字符串
        if isinstance(dynamic_dict, dict):
            for key in dynamic_dict.keys():
                # 如果txt中含有动态词典的每个key的词汇,需将其替换成对应的value
                sub_txt = (
                        '<mstrans:dictionary translation="'
                        + dynamic_dict[key]
                        + '">'
                        + key
                        + "</mstrans:dictionary>"
                )
                text = re.sub(key, sub_txt, text)
            body.append({"text": text})
        elif dynamic_dict == False or dynamic_dict == "":
            body.append({"text": text})
        else:
            print("Neither False nor Dictionary dynamic_dict is !")

    # 当lang_out只有1种语言时,将lang_out变成列表:
    if isinstance(lang_out, str):
        lang_out = [lang_out]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    # response_json = json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
    # print(response_json)

    trans_text = []

    try:
        if isinstance(lang_out, list) and len(lang_out) > 1:
            trans_text = {}
            for i, lang in enumerate(lang_out):
                tmp = []
                for r in response:
                    tmp.append(r["translations"][i]["text"])
                trans_text[lang] = tmp
        else:
            for r in response:
                trans_text.append(r["translations"][0]["text"])
    except:
        print(response)

    return trans_text, response


class Term(BaseModel):
    source: str
    target: str


def Qwen_MT_func(prompt: str, model: str = 'qwen-mt-turbo', api_key: str = None, source_lang: str = 'auto',
                 target_lang: str = 'English', terms: list[Term] = None, tm_list: list[Term] = None,
                 domains: str = None):
    """
    Qwen-MT模型是基于通义千问模型优化的机器翻译大语言模型，擅长中英互译、中文与小语种互译、英文与小语种互译;在多语言互译的基础上，提供术语干预、领域提示、记忆库等能力，提升模型在复杂应用场景下的翻译效果。
    :param prompt: str, 输入的prompt
    :param model: str, 您对翻译质量有较高要求，建议选择qwen-mt-plus模型；如果您希望翻译速度更快或成本更低，建议选择qwen-mt-turbo模型
    :param api_key: str, 阿里云百炼API Key
    :param source_lang: str, 源语言
    :param target_lang: str, 目标语言
    :param terms: list[dict], 技术术语可以提前翻译，并将其提供给Qwen-MT模型作为参考；每个术语是一个JSON对象，包含术语和翻译过的术语信息，格式如下：{"source": "术语", "target": "提前翻译好的术语"}
    :param tm_list: list[dict], 如果您已经有标准的双语句对并且希望大模型在后续翻译时能参考这些标准译文给出结果，可以使用翻译记忆功能；每个JSON对象包含源语句与对应的已翻译的语句，格式如下：{"source": "源语句","target": "已翻译的语句"}
    :param domains: str, 如果您希望翻译的风格更符合某个领域的特性，可以用一段自然语言文本描述您的领域(暂时只支持英文)
    :return: str, 翻译结果
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [{"role": "user", "content": prompt}]

    translation_options = {
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    if terms is not None:
        translation_options['terms'] = terms
    if tm_list is not None:
        translation_options['tm_list'] = tm_list
    if domains is not None:
        translation_options['domains'] = domains

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "translation_options": translation_options
        }
    )
    # print(completion.choices[0].gradio_message.content)
    return completion.choices[0].message.content
