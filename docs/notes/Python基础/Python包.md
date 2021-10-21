# Python 包

在以往自己的一些工作中，常常忽视了一个 Python 包的标准结构，导致最后整个仓库的观感不怎么优雅，模块间调用也麻烦重重，实际上再看 github 上大部分论文的代码仓库，也存在着类似的问题。由于 Python 灵活性很高，这些问题往往都不会导致关键的 bug ，但是为了代码的可读性，还是很有必要规范的。

同时，如果代码仓库需要作为一个标准包上传至 PyPI 之类的索引源并定期更新 ，Python 包的设计规范几乎是必须遵守的。

## 基础知识

现在假设我们有一个这样的文件结构：

```text
sound/                          Top-level package
      __init__.py               Initialize the sound package
      formats/                  Subpackage for file format conversions
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  Subpackage for sound effects
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  Subpackage for filters
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
```

### Python 包是什么？

先给模块做一个定义，一个模块 (Module) 是一个单一的命名空间，其中包括了以下元素：

- 函数 (functions)
- 常量 (constants)
- 类定义 (class definition)
- 内置属性等

通常来说，一个 python 模块对应着一个文件，例如 ```wavread.py```。

包可以是一个或多个模块（甚至是多个包）的组合。包中的模块用 ```包名.模块名``` 表示，以这样一个很简单的方式，避免了不同包中模块的重名混淆问题。

通常来说，一个包对应着一个文件目录。该目录中应该包含 ```__init.py__``` 文件（在 Python 3.3+ 中已不再要求，但为了可读性最好加上）。

### \__init.py__

```__init__.py``` 文件是一个特殊的文件，它相当于名为**该文件父目录名**的包的初始化模块，即如果使用 ```import 父目录名``` 则可以调用在```__init__.py``` 文件中定义或者 ```import``` 的内容。

```__init__.py``` 可以设定在遇到类似 ```import 包名``` 或者 ```from 包名 import *``` 这样很模糊的"全部导入"操作时，究竟该 ```import``` 哪些东西。

例如可以定义内置属性 ```__all__```，来规定在所谓"全部导入"时该引入哪些模块。

```
__all__ = ["echo", "surround", "reverse"]
```

这样就意味着，如果碰到了```from sound.effects import *```这样的引用，应该导入 ```echo```,  ```surround```,  ```reverse``` 三个子模块。

事实上，是否维护```__init.py__```就要看包开发者的心情了，所以**一般情况下应该避免使用** ```import *``` 这样可读性不高的语句。

毕竟，这样引用一个包明显可读性更好：

```
from package import specific_submodule
```

### 模块搜索路径

Python 解释器维护一个列表以供其存放所有可以引入的模块的路径：

```python
import sys
for p in sys.path:
    print p
```

在实际开发中，默认包含了当前目录为搜索路径，所以，当前目录下的模块和子模块均可以正常访问。但是若一个模块**在该模块所在目录**直接运行，同时需要 ```import``` 平级的不同目录的模块，或者上级目录里面的模块，就可以通过修改 ```sys.path``` 这个列表来实现。

不过，在模块所在目录直接运行代码应该尽量避免为好。

## 基本包结构

一个标准的 Python 包结构一般如下：

```python
package_name/
    bin/
    CHANGES.txt
    docs/
    LICENSE.txt
    MANIFEST.in
    README.txt
    setup.py
    package_name/
          __init__.py
          module1.py
          module2.py
          test/
              __init__.py
              test_module1.py
              test_module2.py
```

其中：

`CHANGES.txt`: 记录每次 release 的更改

`LICENSE.txt`: 许可证文件

`MANIFEST.in`: 用于控制文件被包含 / 移出分发包

`README.txt`: README文件

`setup.py`: 包的安装脚本

`bin/`: 顶层脚本存放位置，有些包这个目录命名为 ```script/```

`docs/`: 文档目录

`package_name/`: 包主要代码的存放位置

`test/`: 测试文件目录

## TODO

- ```setup.py``` 的使用
- ```MANIFEST.in``` 的使用

## 参考

https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html

https://packaging.python.org/tutorials/packaging-projects/

https://packaging.python.org/guides/using-manifest-in/
