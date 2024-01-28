| Link                                                                                                    | Model                      | file | size |     |
| ------------------------------------------------------------------------------------------------------- | -------------------------- | ---- | ---- | --- |
| [calypso-3b-alpha-v2-quantificated ](https://huggingface.co/Aryanne/Calypso-3B-alpha-v2-gguf/tree/main) |                            |      |      |     |
| **[EleutherAI/gpt-neo-125M · Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-125M)**            | **～500MB**                |      |      |     |
| [EleutherAI/gpt-neo-1.3B · Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-1.3B)                | ～5GB                      |      |      |     |
| [EleutherAI/gpt-neo-2.7B · Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-2.7B)                | ～10GB                     |      |      |     |
| [EleutherAI/gpt-j-6B · Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B)                        | ～12G(FP16) or ～24G(FP32) |       |     |     |
| [EleutherAI/gpt-neox-20b · Hugging Face](https://huggingface.co/EleutherAI/gpt-neox-20b)                | ～35GB!                    |      |      |     |

### models for sentymental analysis 💌
you can use tow models:

>- 28 emotions
```sh
git lfs install
git clone https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student
```


>- 6 emotions

```sh
https://huggingface.co/nateraw/bert-base-uncased-emotion
```

### 🎉 configure model 🎉

```shell
# git clone https://github.com/jieran233/CyberWaifu.git
# cd CyberWaifu

cd Server

ln -s <path/to/gpt/model/folder> GPT/model
```

## create venv & install pip dependencies 🐍

### termux install

### others

Before create venv, you have to install **python3.10** first. (or using conda environment)

```shell
cd GPT
python3.10 -m venv venv
source venv/bin/activate

# Update
python -m pip install --upgrade setuptools wheel pip

# install pip dependencies
pip install -r requirements.txt
```

####  using pipenv

```shell
pip install pipenv
pipenv install
```


### RUN GPT SERVER 🧠


```shell
# conda deactivate

cd CyberWaifu/Server/GPT
source venv/bin/activate

flask --app api run
# http://127.0.0.1:7210
```

#### using pipenv

```shell
d CyberWaifu/Server/GPT
pipenv shell

flask --app api run
# http://127.0.0.1:5000
```



# Config file Manual

## vector storage database 📑

`CyberWaifu` can use two db motors:

| motor      | description                                                                                                       | modes |
| ---------- | ----------------------------------------------------------------------------------------------------------------- | ----- |
| `chromadb` | it can be used on any pc its very fast but it cannot work's in emulators like termux                              |   support for 3 operation modes `client` `server` `processClient`    |
| `hyperdb`  | this database can be used on emulators like termux that's the principal motive of implement this type of database |      only supports `on premise` mode (it cannot connect to a database in other pc) |



| `"trans-ipt"`  | `"zh"` `"cht"` `"jp"` `"kor"`, etc. or `null` | 翻译输入。调用WebAPI将输入的文本从`指定语言`翻译为英语`en`（使用百度翻译，需在`config/api.json`中填写您的APPID和密钥）https://api.fanyi.baidu.com/doc/21 |
| -------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `"trans-opt"`  | `"zh"` `"cht"` `"jp"` `"kor"`, etc. or `null` | 翻译输出。调用WebAPI翻译生成的文本为`指定语言`（使用百度翻译，需在`config/api.json`中填写您的APPID和密钥）https://api.fanyi.baidu.com/doc/21         |
| `"trans-opt2"` | `"zh"` `"cht"` `"jp"` `"kor"`, etc. or `null` | 第二种语言翻译输出。调用WebAPI翻译生成的文本为`指定语言`（使用百度翻译，需在`config/api.json`中填写您的APPID和密钥）https://api.fanyi.baidu.com/doc/21    |

### config/api.json

appid & key for baidu translator

Ref. [百度翻译开放平台](https://api.fanyi.baidu.com/manage/developer)

## API Manual

请求

```
# GET or POST
# 需要进行百分号转义，别忘了百分号转义的保留字也要转义
http://127.0.0.1:7210/<prompt>

# e.g.
# http://127.0.0.1:7210/%E5%85%B6%E5%AE%9E%EF%BC%8C%E6%88%91%E4%B8%80%E7%9B%B4%E5%96%9C%E6%AC%A2%E7%9D%80%E4%BD%A0%E3%80%82
```

返回

```
# JSON on HTML
# 需要用 html.unescape() 进行HTML反转义

# 为了防止输出内容带有引号导致JSON格式错误，输出内容使用了 base64.urlsafe_b64encode 编码
# 三项分别为raw, 配置的trans-opt, 和配置的trans-opt2 (如果配置为null则不会有那项)

# e.g.
{
    "raw": "IllvdSBhcmUgdGhlIGtpbmQgb2Ygd29tYW4gd2hvIHdvdWxkIHRyeSB0byBtYWtlIG1lIGxvb2sgbGlrZSBJIGFtLiIgSSB0aGVuIHRvbGQgaGVyICJJIHRoaW5rIHlvdSB3b3VsZCBhbHNvIGJlIHdpbGxpbmcgdG8gbGlzdGVuIHRvIG1lLiIgU2hlIHJlcGxpZWQgIkkgdW5kZXJzdGFuZCIsIGFuZCB3ZSBtb3ZlZCBvbnRvICJUb2lsJ3MgbmV3IHBsYWNlLi4udG8gc3BlbmQgb3VyIGRheXMu",
    "zh": "4oCc5L2g5piv6YKj56eN5Lya6K-V5Zu-6K6p5oiR55yL6LW35p2l5YOP5oiR55qE5aWz5Lq644CC4oCd54S25ZCO5oiR5ZGK6K-J5aW54oCc5oiR5oOz5L2g5Lmf5Lya5oS_5oSP5ZCs5oiR55qE44CC4oCd5aW55Zue562U4oCc5oiR55CG6Kej4oCd77yM54S25ZCO5oiR5Lus5pCs5Yiw5LqG4oCc5omY5LyK5bCU55qE5paw5Zyw5pa54oCm4oCm5bqm6L-H5oiR5Lus55qE5pel5a2Q44CC4oCd44CC",
    "jp": "44CM44GC44Gq44Gf44Gv56eB44KS56eB44Gu44KI44GG44Gr6KaL44Gb44KI44GG44Go44GZ44KL5aWz5oCn44Gn44GZ44CN44Gd44GX44Gm56eB44Gv5b285aWz44Gr44CM44GC44Gq44Gf44KC56eB44Gu6KiA44GG44GT44Go44KS6IGe44GE44Gm44GP44KM44KL44Go5oCd44GE44G-44GZ44CN44Go6KiA44Gj44Gf44CC5b285aWz44Gv44CM55CG6Kej44GX44Gm44GE44G-44GZ44CN44Go562U44GI44CB44Gd44GX44Gm56eB44Gf44Gh44Gv44CM44OI44Kk44Or44Gu5paw44GX44GE5aC05omA4oCm4oCm56eB44Gf44Gh44Gu5pel44CF44KS6YGO44GU44GX44G-44GZ44CN44Gr5byV44Gj6LaK44GX44Gf44CC"
}
```

## chroma termux install

the better config is setting in the chroma_db.json file
the follow configuration and 
executing the database in other driver
like in a raspberry pi

```json
{
    "chroma_config": {
        "host": "192.168.15.5",
        "port": 8000,
        "only_http": false
    },
    "top_predictions": 3,
    "current_collection": "",
    "path": "db/",
    "embebingFunction": "all-MiniLM-L6-v2",
    "mode": "server_process"
}
```

then install on termux the pip package `chromadb-client`
using the follow configurations:

`GRPC_PYTHON_DISABLE_LIBC_COMPATIBILITY=1 GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 GRPC_PYTHON_BUILD_SYSTEM_CARES=1 CFLAGS+=" -U__ANDROID_API__ -D__ANDROID_API__=30 -include unistd.h" LDFLAGS+=" -llog" pip install grpcio`
`pip install chromadb-client`
## References

**[EleutherAI/gpt-neo-125M · Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-125M)**

[Transformers Installation](https://huggingface.co/docs/transformers/installation)

[GPT Neo Document](https://huggingface.co/docs/transformers/model_doc/gpt_neo)

[快速上手 &#8212; Flask 中文文档 (2.1.2)](https://dormousehole.readthedocs.io/en/2.1.2/quickstart.html)

[learn-python3/do_flask.py at master · michaelliao/learn-python3 · GitHub](https://github.com/michaelliao/learn-python3/blob/master/samples/web/do_flask.py)

**[GitHub - Animenosekai/translate: A module grouping multiple translation APIs](https://github.com/Animenosekai/translate)**

[Documents - 百度翻译开放平台](https://api.fanyi.baidu.com/doc/21)

[Calypso 3b alpha v2](https://huggingface.co/Xilabs/calypso-3b-alpha-v2)

......and more, thanks.

*I am because we are.*
