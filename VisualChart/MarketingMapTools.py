from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.faker import Faker
from pyecharts.globals import ChartType
from pyecharts.options import GeoRegionsOpts

# from pyecharts import datasets as map_datasets
import json
import pyecharts
from typing import Sequence


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
            label_opts=opts.LabelOpts(is_show=False, font_size=6),
        )
        .add(
            # 系列名称，用于 tooltip 的显示，legend 的图例筛选
            series_name="城市名",
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
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(),
            title_opts=opts.TitleOpts(title=geo_title),
        )
        # .render("geo_guangdong.html")
    )
    return out


if __name__ == "__main__":
    region = "china"

    data_pair = [list(z) for z in zip(Faker.provinces, Faker.provinces)]
    print(data_pair)
    out = geo_gen(region, data_pair, geo_title="测试")
    out.render("geo_guangdong.html")
