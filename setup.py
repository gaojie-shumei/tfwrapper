'''
Created on 2019年9月17日

@author: gaojie-202
'''
import setuptools
# setuptools.setup(
#     name = "tfwrapper",
#     version = "2.0.3",
#     description = "This is a wrapper for tensorflow, the word2vecUtil add method window_format, change datawrapper uncertain length data read",
#     author = "Jie Gao",
#     maintainer = "Jie Gao",
#     author_email = "gaojiexcq@163.com",
#     maintainer_email = "gaojiexcq@163.com",
#     url = "https://github.com/gjxcq/tfwrapper",
#     packages = setuptools.find_packages(exclude=("extendNet")),
#     platforms = "python3",
#     download_url = "https://github.com/gjxcq/tfwrapper",
#     install_requires = [
#         "tensorflow",
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
setuptools.setup(
    name = "tfwrapper-gpu",
    version = "2.0.3",
    description = "This is a wrapper for tensorflow, the word2vecUtil add method window_format, change datawrapper uncertain length data read",
    author = "Jie Gao",
    maintainer = "Jie Gao",
    author_email = "gaojiexcq@163.com",
    maintainer_email = "gaojiexcq@163.com",
    url = "https://github.com/gjxcq/tfwrapper",
    packages = setuptools.find_packages(exclude=("extendNet")),
    platforms = "python3",
    download_url = "https://github.com/gjxcq/tfwrapper",
    install_requires = [
        "tensorflow-gpu",
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