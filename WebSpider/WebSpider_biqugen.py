import requests
import re
import os

def get_response(current_url):
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
    }
    response = requests.get(current_url, headers=header)
    return response

def get_list_url(html_url):
    html_data = get_response(html_url).text
    # 使用正则表达式提取章节链接
    url_list = re.findall('<dd><a href ="(.*?)">', html_data)
    return url_list
    
def get_content(url):
    html_data = get_response(url).text
    # 使用正则表达式提取标题
    title = re.findall('<h1 class="wap_none">(.*?)</h1>', html_data, re.S)[0]
    # 使用正则表达式提取内容
    contents = re.findall('<div id="chaptercontent" class="Readarea ReadAjax_content">(.*?)</div>', html_data, re.S)[0].replace("<br /><br />", "\n").replace("&nbsp;", " ")[93:]
    return title, contents

def save(name, title, contents):
    file = f'novel\\{name}\\'
    if not os.path.exists(file):
        os.mkdir(file)
    with open(file + title + ".txt", 'a', encoding='utf-8') as f:
        f.write("\n\n" + title + "\n\n")
        print(title + " 已保存")
        f.write(contents)
        f.close()

def main(html_url, max=10):
    html_data = get_response(html_url).text
    name = re.findall('<meta property="og:novel:book_name" content="(.*?)"/>', html_data, re.S)[0]
    url_list = get_list_url(html_url)
    for urli in url_list:
        max -= 1
        if max < 0:
            break
        urli = 'https://www.bie5.cc' + urli
        title, contents = get_content(urli)
        save(name, title, contents)

if __name__ == "__main__":
    # html_url = "https://www.bie5.cc/html/45845/"
    html_url = input("请输入网址：")
    main(html_url)