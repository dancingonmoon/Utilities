from pyecharts import options as opts
from pyecharts.charts import Geo, Map
from pyecharts.faker import Faker
from pyecharts.globals import ChartType
from pyecharts.options import GeoRegionsOpts

# from pyecharts import datasets as map_datasets
import json
import pyecharts
from typing import Sequence

params = {
       # 'componentType': 'series',
       # // 系列类型
       # 'seriesType': string,
       # // 系列在传入的 option.series 中的 index
       # 'seriesIndex': 0,
       # // 系列名称
       # 'seriesName': string,
       # // 数据名，类目名
       # 'name': string,
       # // 数据在传入的 data 数组中的 index
       'dataIndex': 0,
       # // 传入的原始数据项
       # 'data': Object,
       # // 传入的数据值
       # 'value': number|Array,
       # // 数据图形的颜色
       # 'color': string,
    }

def geo_gen(
    region: str,
    data_pair: Sequence,
    geo_title: str,
):
    """

    data_pair: 数据项 (坐标点名称，坐标点值)
    :return:
    """
    out = (
        Geo()
        .add_schema(
            maptype=region,
            label_opts=opts.LabelOpts(is_show=False, font_size=10),
            # 地图区域的多边形 图形样式。
            itemstyle_opts=opts.ItemStyleOpts(color=None,border_color=None),
            # 高亮状态下的多边形样式
            emphasis_itemstyle_opts=opts.ItemStyleOpts(),
        )
        .add(
            # 系列名称，用于 tooltip 的显示，legend 的图例筛选
            series_name="被观察的城市",
            # 数据项 (坐标点名称，坐标点值)
            data_pair=data_pair,
            type_=ChartType.EFFECT_SCATTER,
            effect_opts=opts.EffectOpts(
                is_show=True,
            ),
            label_opts=opts.LabelOpts(
                is_show=False,
            ),
        )
        .set_series_opts(label_opts=opts.LabelOpts(
            is_show=False,
            formatter='{b}-{c}'
            # formatter='{params}'
        ))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_show=False,type_='color'),
            title_opts=opts.TitleOpts(title=geo_title),
        )
        # .render("geo_guangdong.html")
    )
    return out

def map_gen(region, data_pair,):


    c = (
        Map()
        .add(
            series_name="商家A",
            data_pair=[list(z) for z in zip(Faker.country, Faker.values())],
            maptype=region,
            label_opts=opts.LabelOpts(is_show=True),
        is_map_symbol_show=False)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Map-世界地图"),
            visualmap_opts=opts.VisualMapOpts(max_=200),
        )
        # .render("map_world.html")
    )
    return c



if __name__ == "__main__":
    region = "world"

    # data_pair = [(x,y) for x,y,z in zip(Faker.provinces, Faker.values(), Faker.values())]
    data_pair = [(x,y) for x,y,z in zip(Faker.country, Faker.values(), Faker.values())]
    # data_pair = [('Canada', 100), ('Brazil', 20), ('Russia', 36), ('United States', 130), ('Africa', 138), ('Germany', 136)]
    print(data_pair)
    # out = geo_gen(region, data_pair, geo_title="测试")
    out = map_gen(region, data_pair,)
    out.render("map_world.html")
    # print(Faker.values())
