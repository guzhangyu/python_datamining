import sys
import urllib,urllib3,base64,json
import random
import logging,datetime,time
# import StringIO
import gzip

# reload(sys)


import re
import requests
# req=requests.get('https://s.taobao.com/list?spm=a217m.8316598.313651-static.4.22b833d5lq0XYU&q=%E9%A3%8E%E8%A1%A3&cat=50344007&style=grid&seller_type=taobao')
# a=req.text
# b=re.match('data-nid="(\\d+)".+?<span class="baoyou-intitle icon-service-free"></span>(.+?)</a>',a)
# print(b.group())

req=requests.get('https://list.tmall.com/search_product.htm?spm=a221t.1710963.8073444875.58.74cb1135iZI8Ie&acm=lb-zebra-7499-292780.1003.4.427990&q=%BA%AB%B0%E6%C3%DE%D2%C2%C4%D0&from=.list.pc_1_searchbutton&type=p&scm=1003.4.lb-zebra-7499-292780.OTHER_14748238221075_427990')
a=req.text
a=a.replace('\r','').replace('\n','')
b=re.findall(r'href="//detail\.tmall\.com/item\.htm\?[^"]*skuId=(\d+)&amp;user\_id=(\d+)[^>]+title="([^"]+)".+?href="(//store\.taobao\.com/search\.htm[^"]+)"',a)
for k in b:
    req=requests.get('https:'+k[3])
    a=req.text
    bi=re.findall(r'class="slogo\-shopname".+?<strong>(.+?)</strong>.+?描 述</div>.+?<span class="shopdsr\-score\-con">([\d\.]+).+?服 务</div>.+?<span class="shopdsr\-score\-con">([\d\.]+).+?物 流</div>.+?<span class="shopdsr\-score\-con">([\d\.]+)',
        a.replace('\n', '').replace('\r', ''))
