# <img alt="BackPACK" src="./logo/backpack_logo_torch.svg" height="90"> BackPACK developer manual

## General standards 
- Python version: support 3.6+, use 3.7 for development
- `git` [branching model](https://nvie.com/posts/a-successful-git-branching-model/)
- Docstring style:  [Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Test runner: [`pytest`](https://docs.pytest.org/en/latest/)
- Formatting: [`black`](https://black.readthedocs.io) ([`black` config](black.toml))
- Linting: [`flake8`](http://flake8.pycqa.org/) ([`flake8` config](.flake8))

---

The development tools are managed using [`make`](https://www.gnu.org/software/make/) as an interface ([`makefile`](makefile)). For an overview, call
```bash
make help 
```
  
## Suggested workflow with [Anaconda](https://docs.anaconda.com/anaconda/install/)
1. Clone the repository. Check out the `development` branch
```bash
git clone https://github.com/f-dangel/backpack.git ~/backpack
cd ~/backpack
git checkout development
```
2. Create `conda` environment `backpack` with the [environment file](.conda_env.yml). It comes with all dependencies installed, and BackPACK installed with the [--editable](http://codumentary.blogspot.com/2014/11/python-tip-of-year-pip-install-editable.html) option. Activate it.
```bash
make conda-env
conda activate backpack
```
3. Install the development dependencies and `pre-commit` hooks
```bash
make install-dev
```
4. **You're set up!** Here are some useful commands for developing
  - Run the tests
    ```bash
    make test
    ```
  - Lint code
    ```bash
    make flake8
    ```
  - Check format (code, imports, and docstrings)
    ```bash
    make format-check
    ```

## Documentation

### Build
- Use `make build-docs`
- To use the RTD theme, uncomment the line `html_theme = "sphinx_rtd_theme"` in `docs/rtd/conf.py` (this line needs to be uncommented for automatic deployment to RTD)

### View
- Go to `docs_src/rtd_output/html`, open `index.html`

### Edit
- Content in `docs_src/rtd/*.rst`
- Docstrings in code
- Examples in `examples/rtd_examples` (compiled automatically)


## Details

- Running quick/extensive tests: ([testing readme](test/readme.md))
- Continuous Integration (CI)/Quality Assurance (QA)
  - [`Travis`](https://travis-ci.org/f-dangel/backpack) ([`Travis` config](.travis.yaml))
    - Run tests: [`pytest`](https://docs.pytest.org/en/latest/)
    - Report test coverage: [`coveralls`](https://coveralls.io)
    - Run examples
  - [`Github workflows`](https://github.com/f-dangel/backpack/actions) ([config](.github/workflows))
    - Check code formatting: [`black`](https://black.readthedocs.io) ([`black` config](black.toml))
    - Lint code: [`flake8`](http://flake8.pycqa.org/) ([`flake8` config](.flake8))
    - Check docstring style: [`pydocstyle`](https://github.com/PyCQA/pydocstyle) ([`pydocstyle` config](.pydocstyle))
    - Check docstring description matches definition: [`darglint`](https://github.com/terrencepreilly/darglint) ([`darglint` config](.darglint))
- Optional [`pre-commit`](https://github.com/pre-commit/pre-commit) hooks [ `pre-commit` config ](.pre-commit-config.yaml)

# Walkthrough: Writing your own extension<a id="sec-1"></a>

- [Walkthrough: Writing your own extension](#sec-1)
  - [Individual gradient pairwise dot products (math details)](#sec-1-1)
  - [Fast-forward: the result](#sec-1-2)
  - [Development process](#sec-1-3)
    - [Extension scaffold](#sec-1-3-1)
    - [Extension implementation](#sec-1-3-2)
    - [Extension test](#sec-1-3-3)
    - [Optional final steps](#sec-1-3-4)

Would you like to write your own [BackPACK](https://www.backpack.pt) extension, but don't know where to start? Don't worry: In this example we will walk you through the full development process of a new extension.

In particular, we will write a first-order extension that computes the pairwise dot product of individual gradients. We will make sure that the extension works properly by writing tests, and mention options to add it to the documentation and example ecosystem on the website.

It's somewhat tricky to illustrate the process in a markdown file, as multiple files are being touched. We tried to split it up into steps with comprehensive `git diff` s and link them below. If you are in a rush, take a look at the full `diff` [here](https://github.com/f-dangel/backpack/compare/9e961ffc8ad454bef215a78188453f0d493ed0ff..da5fde9154242ed0b3667431e15ca2ccc2baf7f2)).

Let's first formulate what we want to end up with.

## Individual gradient pairwise dot products (math details)<a id="sec-1-1"></a>

Let \(\theta \in \mathbb{R}^D\) denote the model parameters, and let the loss \(\mathcal{L}(\theta)\) be a sum or mean of individual losses \(\ell_i(\theta)\),

\begin{align}
  \mathcal{L}(\theta) =
  \begin{cases}
    \frac{1}{N} \sum_{i=1}^N \ell_i(\theta) & \text{(mean)},
    \\
    \sum_{i=1}^N \ell_i(\theta) & \text{(sum)}.
  \end{cases}
\end{align}

The [individual gradients](https://docs.backpack.pt/en/master/extensions.html#backpack.extensions.BatchGrad) \(g_i(\theta) \in \mathbb{R}^D, i = 1, \dots, N\) computed in [BackPACK](https://www.backpack.pt) are

\begin{align}
  g_i(\theta) =
  \begin{cases}
    \frac{1}{N} \nabla_\theta \ell_i(\theta) & \text{(mean)},
    \\
    \nabla_\theta \ell_i(\theta) & \text{(sum)}.
  \end{cases}
\end{align}

We want to compute the pairwise dot products \(g_i^\top g_j\), arranged in an \(N \times N\) matrix \(K\) such that

\begin{align}
  K_{ij}(\theta) = g_i(\theta)^\top g_j(\theta).
\end{align}

## Fast-forward: the result<a id="sec-1-2"></a>

Before we get into the details, let's take a moment to envision how the new extension will be used. Here is a minimal example that should work after the development process.

```python
"""A minimal example how to use the new pairwise gradient extension."""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchDotGrad
from backpack.utils.examples import load_one_batch_mnist

X, y = load_one_batch_mnist(batch_size=4)

model = extend(Sequential(Flatten(), Linear(784, 10),))
lossfunc = extend(CrossEntropyLoss())

loss = lossfunc(model(X), y)
with backpack(BatchDotGrad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".batch_dot.shape:        ", param.batch_dot)
```

The new extension for pairwise individual gradient dot products will be called `backpack.extensions.BatchDotGrad` and provides information about \(K\) through the `batch_dot` attribute.

## Development process<a id="sec-1-3"></a>

In the following, all commands are assumed to be executed from the root directory of the repository.

### Extension scaffold<a id="sec-1-3-1"></a>

Let's run the script and start fixing first problems.

```bash
git checkout c2212d5d5c6b30f4f449dd926d8a057966e1cb79
python new_extension/grad_dot.py
git checkout -
```

Output:

```bash
Traceback (most recent call last):
File "new_extension/grad_dot.py", line 6, in <module>
  from backpack.extensions import BatchDotGrad
ImportError: cannot import name 'BatchDotGrad' from 'backpack.extensions' (...)
```

Unsurprisingly, we encounter missing imports. [BackPACK](https://www.backpack.pt) does not know yet about the new extension, so let's create dummy modules (see [this commit](https://github.com/f-dangel/backpack/commit/a4bdf3a8ff7ed6eeafb800943811f37204f7a99e)):

-   First-order extensions reside in the `backpack/extensions/firstorder/` directory. Inside it, create a new directory `batch_dot_grad` with an `__init__.py` file. This file contains a dummy of the new `BatchDotGrad` extension.
-   To be able to import `BatchDotGrad` from `backpack.extensions`, add it in the `__init__.py` files of `backpack/extensions/firstorder` and `backpack/extensions`.

Let's re-run the code. Imports should be resolved:

```bash
git checkout 2a8ccad1d0bd3268abe2b3609756b3bf57fb4e0a
python new_extension/grad_dot.py
git checkout -
```

Output:

```bash
Traceback (most recent call last):
  File "new_extension/grad_dot.py", line 21, in <module>
    print(".batch_dot.shape:        ", param.batch_dot)
AttributeError: 'Parameter' object has no attribute 'batch_dot'
```

The code passes the `import` statements and downloads MNIST, but then complains about a missing attribute in one of the model parameters.

To get a better understanding what is going on, enable [BackPACK](https://www.backpack.pt)'s `debug` mode (see [this commit](https://github.com/f-dangel/backpack/commit/8e3413d1782c1dad041215b3241c6f998a6364a9)):

```bash
git checkout 8e3413d1782c1dad041215b3241c6f998a6364a9
python new_extension/grad_dot.py
git checkout -
```

Output (only the important parts):

```bash
[DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on CrossEntropyLoss()
[DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on Linear(in_features=784, out_features=10, bias=True)
[DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on Sequential(
  (0): Flatten()
  (1): Linear(in_features=784, out_features=10, bias=True)
)
```

While the The dummy extension is being executed during the `backward` pass, it does not perform any action that initializes the `batch_dot` attribute.

### Extension implementation<a id="sec-1-3-2"></a>

We have to implement the action that should be carried out by an `extend`-ed layer during a `backward` that requests the `BatchDotGrad` quantity. By default, nothing is done.

1.  Module extension scaffold

    Note that only layers with parameters need to compute the pairwise dot products. For our particular network, this means that we have to cover what should be done by `torch.nn.Linear`.
    
    Here are the steps (see [this commit](https://github.com/f-dangel/backpack/commit/caddf88f51f12a43ecfe282690fc705fcc2d7469)) to set up a dummy:
    
    -   Create a `BatchDotGradLinear` module extension with functions `bias` and `weight`. They will be executed as backward hooks and their output is stored in the `bias.batch_dot` and `weight.batch_dot` attributes.
    -   Inform the `BatchDotGrad` extension about the module-to-backward-hook-mapping between `torch.nn.Linear` and `BatchDotGradLinear` by adding an entry in the `module_exts` dictionary.
    
    To keep things simple for the moment, this dummy always 'computes' a value of `42`, and prints a message to the command line:
    
    ```bash
    git checkout caddf88f51f12a43ecfe282690fc705fcc2d7469
    python new_extension/grad_dot.py
    git checkout -
    ```
    
    Output:
    
    ```bash
    [DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on CrossEntropyLoss()
    [DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on Linear(in_features=784, out_features=10, bias=True)
    Executing BatchDotGradLinear for bias (return dummy value of 42)
    Executing BatchDotGradLinear for weight (return dummy value of 42)
    [DEBUG] Running extension <backpack.extensions.firstorder.batch_dot_grad.BatchDotGrad object at ...> on Sequential(
      (0): Flatten()
      (1): Linear(in_features=784, out_features=10, bias=True)
    )
    1.weight
    .grad.shape:              torch.Size([10, 784])
    .batch_dot.shape:         42
    1.bias
    .grad.shape:              torch.Size([10])
    .batch_dot.shape:         42
    ```
    
    We're almost there! Let's implement the math.

2.  Module extension implementation

    To compute pairwise individual gradient dot products, we need to compute the individual gradients and take their dot products during a backward pass.
    
    For the first step, we can borrow the functionality from [BackPACK](https://www.backpack.pt)'s [`BatchGrad`](https://docs.backpack.pt/en/master/extensions.html#backpack.extensions.BatchGrad) extension (from \`backpack.core\`). Pairwise dot products are obtained with `torch.einsum` magic (see [this commit](https://github.com/f-dangel/backpack/commit/157e9a5d3ffb4d0e6e5727aa65811fa67bcd0b63)).
    
    Now, the code should work and we can disable `debug`.
    
    ```bash
    git checkout 157e9a5d3ffb4d0e6e5727aa65811fa67bcd0b63
    python new_extension/grad_dot.py
    git checkout -
    ```
    
    Output:
    
    ```:eval
    1.weight
    .grad.shape:              torch.Size([10, 784])
    .batch_dot.shape:         torch.Size([4, 4])
    1.bias
    .grad.shape:              torch.Size([10])
    .batch_dot.shape:         torch.Size([4, 4])
    ```
    
    Indeed, the pairwise individual gradients stored in `batch_dot` and have the expected shape.
    
    That's it for the functionality part of our extension, but we're not completely done yet.

### Extension test<a id="sec-1-3-3"></a>

Untested code is useless code. To verify the results obtained by our new extension, we have to write tests. [BackPACK](https://www.backpack.pt) has a test suite which we will use. It aims to take off as much work as possible. In its folder structure, it follows the main library.

Once we've completed the steps below, testing our extension under a new neural network setting only requires to specify an entry in a dictionary.

The rough steps are:

-   Set up a scaffold
-   Implement the computation of pairwise dot products in `torch.autograd`. Add the `backpack` implementation of pairwise individual gradients.
-   Specify test settings

1.  Test suite scaffold

    Similar to the files we created while writing the functionality for `BatchDotGrad`, the same should be done in the test suite (see [this commit](https://github.com/f-dangel/backpack/commit/df8cacd6d3b62c913340f39689559d9eb77aef78)):
    
    -   Tests of first-order extensions reside in `test/extensions/firstorder/`. In that folder create a new `/atch_dot_grad` directory with the following files:
        1.  `__init__.py`: Leave empty.
        2.  `batch_dot_grad_settings.py`: Configuration file for specifying the settings that should be tested.
        3.  `test_batch_dot_grad.py`: Contains the test function that will be run by `pytest`.
    
    The test function for `BatchDotGrad` is called `test_batch_dot_grad`. We can restrict the test suite to run only that function with the `pytest`'s `-k` flag:
    
    ```bash
    git checkout df8cacd6d3b62c913340f39689559d9eb77aef78
    pytest -vx . -k batch_dot_grad
    git checkout -
    ```
    
    Output (only relevant content):
    
    ```bash
    test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py::test_batch_dot_grad[example-dev=cpu-in=(3,10)-model=Sequential(\n(0):Linear(in_features=10,out_features=5,bias=True)\n)-loss=CrossEntropyLoss()] PASSED                                                                                         [100%]
    ```
    
    A single test got executed, and passed. The 'test' performed in `test_batch_dot_grad` is a dummy that needs to be replaced with the real comparison we want to make.

2.  Comparison with `torch.autograd`

    In the test, the pairwise dot products will be computed via `torch.autograd` and with `backpack`, then compared.
    
    Implementations with both approaches are in the `test/extensions/implementation` directory. We have to extend them with functions that compute the pairwise individual gradient dot products (see [this commit](https://github.com/f-dangel/backpack/commit/204a4df674f5a38e7a55df6954b2d8be8565be3e)).
    
    These implementations can now be called, and their results be compared, by the test function (see [this commit](https://github.com/f-dangel/backpack/commit/2ae15e3aec095737359e401a3aff0f0dcb3063ce)):
    
    ```bash
    git checkout 2ae15e3aec095737359e401a3aff0f0dcb3063ce
    pytest -vx . -k batch_dot_grad
    git checkout -
    ```
    
    Output (only relevant conten):
    
    ```bash
    test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py::test_batch_dot_grad[example-dev=cpu-in=(3,10)-model=Sequential(\n(0):Linear(in_features=10,out_features=5,bias=True)\n)-loss=CrossEntropyLoss()] PASSED                                                                                         [100%]
    ```
    
    Nice! The non-dummy test passes and we can be more confident that our extension does its job properly.

3.  Specifying more settings

    Let's add a new test setting for `BatchDotGrad` to increase test coverage. We only have to touch the settings file `test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py`, that also contains documentation how to add a new entry.
    
    [This commit](https://github.com/f-dangel/backpack/commit/c420be48df3e73e0b7b47f5e1f1a8eb2c66e3a04) adds a small neural net with sigmoid activations to the settings. Re-running the test suite, we can see that there are now more tests being executed (and even on multiple devices if your machine has a GPU):
    
    ```bash
    git checkout c420be48df3e73e0b7b47f5e1f1a8eb2c66e3a04
    pytest -vx . -k batch_dot_grad
    git checkout -
    ```
    
    Output (on a GPU machine):
    
    ```bash
    test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py::test_batch_dot_grad[example-dev=cpu-in=(3,10)-model=Sequential(\n(0):Linear(in_features=10,out_features=5,bias=True)\n)-loss=CrossEntropyLoss()] PASSED                                                                                         [ 33%]
    test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py::test_batch_dot_grad[dev=cpu-in=(4,10)-model=Sequential(\n(0):Linear(in_features=10,out_features=5,bias=True)\n(1):Sigmoid()\n(2):Linear(in_features=5,out_features=3,bias=True)\n)-loss=CrossEntropyLoss()] PASSED                              [ 66%]
    test/extensions/firstorder/batch_dot_grad/test_batch_dot_grad.py::test_batch_dot_grad[dev=cuda-in=(4,10)-model=Sequential(\n(0):Linear(in_features=10,out_features=5,bias=True)\n(1):Sigmoid()\n(2):Linear(in_features=5,out_features=3,bias=True)\n)-loss=CrossEntropyLoss()] PASSED                             [100%]
    ```
    
    Congratulations if you made it up to this point! You've successfully developed your own [BackPACK](https://www.backpack.pt) extension. Continue reading if you're interested how to make it more visible and easier to use for other.

### Optional final steps<a id="sec-1-3-4"></a>

1.  Add extension documentation

    For adding your extension to the documentation on the website, add it to the `docs_src/rtd/extensions.rst` file (see [this commit](https://github.com/f-dangel/backpack/commit/15bfc65c3c9a2d6db70f326751365dbeb03433bf)).
    
    You can run `make build-docs` and inspect `docs_src/rtd_output/extensions.html` locally for a preview.

2.  Add extension to all-in-one example

    It's easier for others to figure out how your extension is used if you provde an an example. The [all-in-one](https://docs.backpack.pt/en/master/basic_usage/example_all_in_one.html) example summarizes all extensions provided by [BackPACK](https://www.backpack.pt) and is the best place to present your extension (see [this commit](https://github.com/f-dangel/backpack/commit/da5fde9154242ed0b3667431e15ca2ccc2baf7f2)).


###### _BackPACK is not endorsed by or affiliated with Facebook, Inc. PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc._
