from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="wavtokenizer",
    version="1.0.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="Discrete Codec Models With Forty Tokens Per Second for Audio Language Modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shengpeng Ji, Ziyue Jiang, Xize Cheng, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, Wen Wang, Zhou Zhao",
    url="https://github.com/jishengpeng/WavTokenizer",
    license="MIT",
    packages=find_packages(),
    keywords=["audio", "compression", "machine learning"],
)
