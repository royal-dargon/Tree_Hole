import requests
import re
import json
import lxml

base_url = "https://m.weibo.cn/api/container/getIndex?containerid=100808bf27711bf93d6da034ece4fac96b8b40&luicode"\
        "=10000011&lfid=100103type%3D1%26q%3D%E6%A0%91%E6%B4%9E&since_id="


def main():
    requests.packages.urllib3.disable_warnings()
    url = "https://m.weibo.cn/api/container/getIndex?containerid=100808bf27711bf93d6da034ece4fac96b8b40&luicode" \
          "=10000011&lfid=100103type%3D1%26q%3D%E6%A0%91%E6%B4%9E"
    # 设置需要爬取的页数
    n = 20
    count = 0
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
                        blog_texts = text.get('text')
                        pattern = re.compile("[\u4e00-\u9fa5]+")
                        blog_text = re.findall(pattern, blog_texts)
                        user = text.get('user')
                        user_id = user.get('id')
                        name = user.get('screen_name')
                        description = user.get('description')
                        location = text.get('region_name')
                        new_text = ""
                        for k in range(len(blog_text)):
                            if len(blog_text[k]) <= 1:
                                continue
                            if k != 0:
                                new_text += str(blog_text[k])
                                if k < len(blog_text)-1:
                                    new_text += ","
                                else:
                                    new_text += "。"
                            else:
                                if blog_text[k] != "树洞":
                                    new_text += str(blog_text[k])
                        blog_info = [create_time, new_text, user_id, name, description, location]
                        print(blog_info)
                        count += 1
    print("共爬取了%d条博客" % count)


if __name__ == "__main__":
    main()
