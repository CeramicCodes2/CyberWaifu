# CyberWaifu

this folk implements updates in the GPT server and the Live2d modul


## folk objectives:

- [x] a vector storage database for long conversations what can be personalized for server in local mode, at a diferent process or as a http server
- [x] a sumarization implementation
- [x] new configuration system what allow personalization of your waifu
- [ ] a support for different backends like gpt4all,llama.cpp and text generation webui
- [ ] implement sentimental analysis
- [x] better implementation for live2d


## Introduction

[CyberWaifu Server](https://github.com/jieran233/CyberWaifu/blob/main/Server) = [Flask](https://flask.palletsprojects.com) + [ ( [GPT](https://github.com/jieran233/CyberWaifu) + [Online translation](https://github.com/Animenosekai/translate) ) + [Tacotron2 / VITS](https://github.com/luoyily/MoeTTS) + [Live2D](https://github.com/jieran233/CyberWaifu/blob/main/Server/Live2D) ]

[CyberWaifu Client](https://github.com/jieran233/CyberWaifu/blob/main/Client) = [PyQt6](https://pypi.org/project/PyQt6/) + ( [PyQt6-WebEngine](https://pypi.org/project/PyQt6-WebEngine/) + [Requests](https://requests.readthedocs.io/) )

![Screenshot_20221230_185828.png](https://s2.loli.net/2022/12/30/qBkD4s5wIOdLhgS.png)

## Read more

>- [./Server/GPT/README.md](https://github.com/CeramicCodes2/CyberWaifu/tree/main/Server/README.md)
>- [./Server/TTS/README.md](https://github.com/jieran233/CyberWaifu/blob/main/Server/TTS/README.md)
>- [./Server/Live2D/README.md](https://github.com/jieran233/CyberWaifu/blob/main/Server/Live2D/README.md)
>- [./Server/BetaLive/README.md](https://github.com/CeramicCodes2/CyberWaifu/blob/main/Server/BetaLive/readme.md)
>- [./Client/README.md](https://github.com/jieran233/CyberWaifu/blob/main/Client/README.md)
