# *This file is still work in progress!*

## Development Requirements
* gitpython 
* unittest (built-in)

## Coding formatting
* Follow the [pep8](https://www.python.org/dev/peps/pep-0008/) guidelines and always format before submitting pull request    

#### No single letter variable names unless it's obvious. for ex.
### Good
```
N = len(data)
```

### Bad
```
T = func.run_transformation()
```

### Docstrings
* Create docstrings for all classes and functions
* Use `DESCRIPTION`, `ARGS`, `RETURNS` headers
* Indent after each header message
* Argument names must have a colon before and after name `:argname: a description`
```

class Pipeline:
    """
    DESCRIPTION:
        This class is the bridge between running features on images.
        Automates the process of extracting each feature, saving it,
        outputing it and putting it in dictionary form

    ARGS:
        :data: image data
        :ops: features to run on images. These ops don't have dependencies
        :seq: features to run in sequential way (output is input to another)
        :save_all: boolean check to save all features ran
        :models: dictionary of statistical models to run on the data
    """
    def __init__(self, data=None, ops=None, seq=None, save_all=None,
                 models=None):
        ...

    def pad_batch(batch, batch_size, frame, pad=True):
        """
        DESCRIPTION:
            Take in a batch, pad it with 0s if necessary
            and return appended batch

        ARGS:
            :frame: a numpy array extracted from an MPEG
                    (length x width x channel)
            :batch: list of frames
            :batch_size: number of frames per batch (integer >= 1)

        RETURNS:
            batch: list of a batch, batch has batch_size frames,
            each an ndarray of shape (L x W x C)
        """
```

#### General good coding practices that's relevant [Link](https://gist.github.com/sloria/7001839)


## Test style
* All subfolders must start with `test_` followed byt the module where the class you writing test for is in, for ex. `test_decode` 
* Use CamelCase for test class names - `TestingClass:`
* test functions must also start with `_test`

```
class TestName(unittest.TestCase):

    def setUp(self):
        ...
    def tearDown(self):
        ...

    def test_function_name(self):
        ...
```

## Running Testing
change directory to directory of test folder:
```
cd /path/to/repo/image_analysis/image_analysis
```

Run unittest discover
```
python -m unittest discover
```

Run a single test case
```
python -m unittest test.module_name.test_file_name
```

Or for a specific function
```
python -m unittest test.module_name.test_file_name.test_class_name.test_function
```

## Building docs
Switch to ```gh-pages``` branch and run gen-docs.py
```
git checkout gh-pages
python gen-docs.py
```

## Submitting Code
* Create your own [Branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/), *Never Work on master directly*
* Create a [Pull Request](https://help.github.com/articles/about-pull-requests/)
* Wait for code review! and tackle your next bug! :sunglasses: