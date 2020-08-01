'''
Created on 2019年9月17日

@author: gaojie-202
'''
import setuptools
description = '''
    This is a wrapper for tensorflow and a word2vecUtil with gensim, 
    this version rewrite the datawrapper for general and set it in basev2 package,
    or can use it with shuffix 'V2'
    '''
setuptools.setup(
    name = "tfwrapper",
    version = "2.1.0",
    description = description,
    author = "Jie Gao",
    maintainer = "Jie Gao",
    author_email = "gaojiexcq@163.com",
    maintainer_email = "gaojiexcq@163.com",
    url = "https://github.com/gaojie-shumei/tfwrapper",
    packages = setuptools.find_packages(exclude=("extendNet")),
    platforms = "python3",
    download_url = "https://github.com/gaojie-shumei/tfwrapper",
    install_requires = [
        "tensorflow>=1.13.1",
        "gensim<=3.6.0",
        "numpy"
    ],
    requires = [
        "tensorflow",
        "gensim",
        "numpy"
    ],
    license = "MIT"
)
# setuptools.setup(
#     name = "tfwrapper-gpu",
#     version = "2.1.0",
#     description = description,
#     author = "Jie Gao",
#     maintainer = "Jie Gao",
#     author_email = "gaojiexcq@163.com",
#     maintainer_email = "gaojiexcq@163.com",
#     url = "https://github.com/gaojie-shumei/tfwrapper",
#     packages = setuptools.find_packages(exclude=("extendNet")),
#     platforms = "python3",
#     download_url = "https://github.com/gaojie-shumei/tfwrapper",
#     install_requires = [
#         "tensorflow-gpu>=1.13.1",
#         "gensim<=3.6.0",
#         "numpy"
#     ],
#     requires = [
#         "tensorflow",
#         "gensim",
#         "numpy"
#     ],
#     license = "MIT"
# )
