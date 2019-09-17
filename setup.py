'''
Created on 2019年9月17日

@author: gaojie-202
'''
import setuptools
setuptools.setup(
    name = "tfwrapper",
    version = "1.0",
    description = "This is a wrapper for tf.Session().run and tf.data.Dataset in tensorflow1.x",
    author = "Jie Gao",
    maintainer = "Jie Gao",
    author_email = "gaojiexcq@163.com",
    maintainer_email = "gaojiexcq@163.com",
    url = "https://github.com/gjxcq/tfwrapper",
    packages = setuptools.find_packages(exclude=("extendNet")),
    platforms = "python3",
    download_url = "https://github.com/gjxcq/tfwrapper",
    install_requires = [
        "tensorflow==1.13.1",
        "gensim==3.6.0",
        "numpy==1.16.4"
    ],
    requires = [
        "tensorflow",
        "gensim",
        "numpy"
    ]
)