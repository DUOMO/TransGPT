#!/usr/bin/env python
# coding: utf-8

"""
@author: WJ
@software: PyCharm
@time: 2023/7/17 15:59
"""
# # 调用高德api Web服务里的模块功能
# ## 地理编码


import requests
key_private = 'ba7c608dccdfcbbec50441ddb88466e3'

# 地理编码
def geo(add:str,city:str)-> dict:
    """获取地理编码"""
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    param ={
        'key':key_private,
        'address':add,
        'city':city,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# geo('广东省广州市从化区中山大学南方学院','广州')


# ## 逆地理编码

# In[3]:


def regeo(location:str)->dict:
    """获取逆地理编码"""
    url ='https://restapi.amap.com/v3/geocode/regeo?parameters'
    param ={
        'key':key_private,
        'location':location,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# regeo('113.679287,23.632575')


# ## 路径规划

# In[4]:


def walking(origin:str,destination:str)->dict:
    """步行路径规划"""
    url = 'https://restapi.amap.com/v3/direction/walking?parameters'
    param = {
        'key':key_private,
        'origin':origin,
        'destination':destination,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# walking('116.481028,39.989643','116.434446,39.90816')


# ## 行政区域查询

# In[5]:


def config_district(keywords:str)->dict:
    url='https://restapi.amap.com/v3/config/district?parameters'
    param ={
        'key':key_private,
        'keywords':keywords,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# config_district('广州')


# ## 搜索POI

# In[6]:


def place(keywords:str,types:str)->dict:
    url='https://restapi.amap.com/v3/place/text?parameters'
    param ={
        'key':key_private,
        'keywords':keywords,
        'types':types,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# place('广州','酒店')


# ## IP定位

# In[7]:


def ip(ip:str)->dict:
    url='https://restapi.amap.com/v3/ip?parameters'
    param ={
        'key':key_private,
        'ip':ip,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# ip('61.242.54.251')


# ## 静态地图

# In[8]:


from PIL import Image
from io import BytesIO
def staticmap(location:str)->dict:
    url='https://restapi.amap.com/v3/staticmap?parameters'
    param ={
        'key':key_private,
        'location':location,
        'zoom':"15",
        'size':'1920*1080'       
    }
    response = requests.get(url,params=param)
    data = Image.open(BytesIO(response.content))
    return data
# staticmap('113.291418,23.094611')


# ## 坐标转换

# In[9]:


def coordinate(locations:str)->dict:
    url='https://restapi.amap.com/v3/assistant/coordinate/convert?parameters'
    param ={
        'key':key_private,
        'locations':locations,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# coordinate('23.081018,113.153374')


# ## 天气查询

# In[10]:


def weather(city:str,extensions:str)->dict:
    url='https://restapi.amap.com/v3/weather/weatherInfo?parameters'
    param ={
        'key':key_private,
        'city':city,
        'extensions':extensions,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# weather('黑龙江','all')


# ## 输入提示

# In[11]:


def assistant(keywords:str)->dict:
    url='https://restapi.amap.com/v3/assistant/inputtips?parameters'
    param={
        'key':key_private,
        'keywords':keywords,
        'output':'json'
    }
    response = requests.get(url,params=param)
    data = response.json()
    return data
# assistant('芝士')


# ## 交通态势

# In[12]:


def traffic(level:str,rectangle:str)->dict:
    url='https://restapi.amap.com/v3/traffic/status/rectangle?parameters'
    param={
        'key':key_private,
        'level':level,
        'rectangle':rectangle,
        'output':'json'
    }
    response =requests.get(url,params=param)
    data = response.json()
    return data
# traffic('2','116.351147,39.966309;116.357134,39.968727')


# ## 地理围栏

# In[16]:


def geofence():
    """创建围栏"""
    url='http://restapi.amap.com/v4/geofence/meta?key=你申请的key'
    params={
        'name':'围栏名称',
        'center':'113.293046,23.103682',
        'radius':'3000',
        'enable':'true',
        'repeat':'Mon,Tues,Wed,Thur,Fri,Sat,Sun',
        'output':'json'
    }
    data =requests.post(url,json=params).json()
    return data
# geofence()


# ## 轨迹纠偏

# In[17]:


def grasproad()->dict:
    url='https://restapi.amap.com/v4/grasproad/driving'
    param={
        'key':key_private,
        "x": 116.449429,
        "y": 40.014844,
        "sp": 4,
        "ag": 110,
        "tm": 1478831753,
        'output':'json'
        
    }
    response = requests.post(url,params=param)
    data = response.json()
    return data
# grasproad()

if __name__ == '__main__':
    # # 地理编码
    # geo('广东省广州市从化区中山大学南方学院', '广州')
    # # 逆地理编码
    # regeo('113.679287,23.632575')
    # # 路径规划
    # walking('116.481028,39.989643', '116.434446,39.90816')
    # # 行政区域查询
    # config_district('广州')
    # # 搜索POI
    # place('广州', '酒店')
    # ##IP 定位
    # ip('61.242.54.251')
    # # 静态地图
    # staticmap('113.291418,23.094611')
    # # 坐标转换
    # coordinate('23.081018,113.153374')
    # # 输入提示
    data = assistant('芝士')
    print(data)
    # # 交通态势
    # traffic('2', '116.351147,39.966309;116.357134,39.968727')
    # # 地理围栏
    # geofence()
    # # 轨迹纠偏
    # grasproad()





