from urllib import request
import re
import city

class map_query():
    url = 'https://tianqi.so.com/weather/'
    # 包含天气日期星期的html代码段
    weather_week = '<ul class="weather-columns"><li>([\s\S]*?)</li'
    # 日期与星期
    week = '<!-- ([\S\s]*?) -->'
    # 日期与星期
    dataaa = '-->([\s\S]*?)</div>'
    # 包含天气的代码段
    weather =  r'<div class="weather-icon weather-icon-.*\n\s*[\u4e00-\u9fa5]*\n\s*</div><div>' #r'<div class="weather-icon weather-icon-([\s\S]*?)\n'

    # 包含指数的html的代码段
    index_data = r'<div class="tab-pane"([\S\s]*?)</div></div></div>'
    # 指数
    index_index = '<div class="tip-title tip-icon-([\S\s]*?)">'
    # 建议
    index_sugge = '<div class="tip-cont" title="([\s\S]*?)"'

    # 城市名称
    city_n = ""

    # 模拟http请求（私有方法）
    def __get_htmls(self, codes):
        urll = str(self.url + codes)
        ht = request.urlopen(urll)
        html = ht.read()
        html = str(html, encoding='utf-8')

        return html

    def __analyze(self, ff, html):
        # 天气
        if ff == 1:
            weather_weeks = re.findall(self.weather_week, html)
        # 指数
        else:
            weather_weeks = re.findall(self.index_data, html)

        return weather_weeks

    # 查询近期十五天之内的天气
    def __analyze_weather(self, weather_weeks):

        star_lists = []

        for we in weather_weeks:
            data_l = re.findall(self.dataaa, we)  # 分割星期，
            weather_l = re.findall(self.weather, we)  # 天气

            data_ll = data_l[0].split()
            week1 = data_ll[0]  # 星期
            data1 = data_ll[1]  # 日期
            weat = weather_l[0].split()
            weather_l = weat[3]  # 天气

            star_list = {'week': week1, "data": data1, "weather": weather_l}
            star_lists.append(star_list)

        return star_lists

    def __analyze_indexs(self, index_code):

        star_indexs = []

        for ind in index_code:
            indexs = re.findall(self.index_index, ind)
            sugges = re.findall(self.index_sugge, ind)

            for i in (range(0, len(indexs) - 1)):
                res_index = indexs[i].split('"')
                star_index = {'index': res_index[2], 'sugges': sugges[i]}
                star_indexs.append(star_index)

        return star_indexs

    def __show_weather(self, star_lists):
        print("***************** %s近十五天的天气如下：*************\n\n" % self.city_n)
        for re in star_lists:
            print("星期：" + re['week'] + "    日期：" + re['data'] + "   天气：" + re['weather'])
            # print(rs)

    def __show_index(self, star_indexs):
        print("*****************  两天指数及建议： ****************\n")
        print("\n\n*****************  今天建议如下  *****************\n\n")
        l = 0
        for re in star_indexs:
            l = l + 1
            fg = re['index'].split("：")

            if fg[0] == "过敏指数":
                if l > 1:
                    print("\n\n*****************  明天建议如下  *****************\n\n")
            print(re['index'] + "\t         建议：" + re['sugges'])

    def __city_num(self):

        city_name = input("请输入城市名：")
        code = city.citycode[city_name]
        self.city_n = city_name
        print("您查询的是" + city_name + "城市代码为：" + code)

        if not city_name in city.citycode:
            print("这个城市不存在！")
            exit()
            code = city.citycode[city_name]
        return code

    def go(self):
        codes = self.__city_num()
        html = self.__get_htmls(codes)

        # 查询天气
        weather_weeks = self.__analyze(1, html)
        star_lists = self.__analyze_weather(weather_weeks)
        # 查询指数
        indexs = self.__analyze(2, html)
        star_indexs = self.__analyze_indexs(indexs)

        # 展示天气
        self.__show_weather(star_lists)

        # 展示指数
        self.__show_index(star_indexs)

if __name__ == '__main__':
    mp = map_query()
    mp.go('北京朝阳门', '')