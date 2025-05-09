import requests
import re
import json
import os

def get_response(url):
    headers = {
        'referer': 'https://www.bilibili.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'
    }
    response = requests.get(url=url, headers=headers, proxies={})
    return response

def get_video_content(url):
    response = get_response(url)
    info = re.findall('<script>window.__playinfo__=(.*?)</script>', response.text)[0]
    json_data = json.loads(info)

    audio_url = json_data['data']['dash']['audio'][0]['baseUrl']
    video_url = json_data['data']['dash']['video'][0]['baseUrl']
    title = re.findall('<h1 data-title="(.*?)" title', response.text)[0].replace(' ', '')

    return audio_url, video_url, title

def Save(title, audio_url, video_url):
    file = f'video\\'
    if not os.path.exists(file):
        os.mkdir(file)
    audio_content = get_response(audio_url).content
    video_content = get_response(video_url).content
    with open(f'video\\{title}.mp3', 'wb') as f:
        f.write(audio_content)
    print(f'音频文件 {title}.mp3 已保存')
    with open(f'video\\{title}.mp4', 'wb') as f:
        f.write(video_content)
    print(f'视频文件 {title}.mp4 已保存')
    res = f"bilibili视频爬取完成，文件名为 './video/{title}.mp4'"
    return res

def main(url):
    audio_url, video_url, title = get_video_content(url)
    res = Save(title, audio_url, video_url)
    return res

if __name__ == "__main__":
    # url = 'https://www.bilibili.com/video/BV1Ak4y1n74o'
    html_url = input("请输入网址：")
    main(html_url)