import requests
from pyquery import PyQuery as pq
import openpyxl as op
import urllib
import datetime
import time

base_url = "https://m.weibo.cn/api/container/getIndex?containerid=100808bf27711bf93d6da034ece4fac96b8b40&luicode"\
        "=10000011&lfid=100103type%3D1%26q%3D%E6%A0%91%E6%B4%9E&since_id="


# download the pictures
def download_pic(pic_url, pic_id):
    pic_path = "img" + '\\'
    f = open(pic_path + str(pic_id) + ".jpg", 'wb')
    f.write((urllib.request.urlopen(pic_url)).read())
    f.close()
    time.sleep(0.1)  # 下载间隙


def main():
    start_time = time.time()
    wb = op.load_workbook("test.xlsx")
    # 创建子表
    ws = wb['info']
    requests.packages.urllib3.disable_warnings()
    url = "https://m.weibo.cn/api/container/getIndex?containerid=100808bf27711bf93d6da034ece4fac96b8b40&luicode" \
          "=10000011&lfid=100103type%3D1%26q%3D%E6%A0%91%E6%B4%9E"
    # 设置需要爬取的页数
    n = 100
    count = 0
    img_count = 0
    # 创建表头
    ws.append(['id', 'time', 'text', 'uid', 'uname', 'description', 'location'])
    for i in range(n):
        result = requests.get(url, verify=False)
        if result.status_code != 200:
            print("爬取出现问题！再次尝试！")
            continue
        else:
            contents = result.json()
            # 这里是根据找到的特点更新下一页要爬取的url
            since_id = contents.get('data').get('pageInfo')['since_id']
            url = base_url + str(since_id)
            # 下面开始获取正文部分的内容
            details = contents.get('data').get('cards')
            for j in range(len(details)):
                if "card_group" in details[j].keys():
                    text = details[j].get("card_group")[0]
                    if "mblog" in text.keys():
                        text = text.get('mblog')
                        create_time = text.get('created_at')
                        blog_text = pq(text.get("text")).text()
                        user = text.get('user')
                        user_id = user.get('id')
                        name = user.get('screen_name')
                        description = user.get('description')
                        location = text.get('region_name')
                        pics = text.get('pics')
                        if pics:
                            for pic in pics:
                                pic_url = pic.get('large').get('url')
                                download_pic(pic_url, count)
                        # if len(img) != 0:
                        #     img_count += 1
                        blog_info = [count+1, create_time, blog_text, user_id, name, description, location]
                        ws.append(blog_info)
                        count += 1
    wb.save("test.xlsx")
    end_time = time.time()
    act_time = end_time - start_time
    print("共爬取了%d条博客" % count)
    print("程序运行的时间为%d" % act_time)


if __name__ == "__main__":
    main()
