### Setting up your environment for development

1. Fork the repository from github and clone your fork locally (see [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to setup your ssh key):

```bash
git clone git@github.com:<your_username>/HalfPipe2Bids.git
```

2. Set up a virtual environment to work in using whichever environment management tool you're used to and activate it. If you don't have one, we recommend [`uv`](https://docs.astral.sh/uv/pip/environments/). We are using Python >=3.11.

Minimal example:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

With `uv`:

```bash
cd HalfPipe2Bids
uv venv
source .venv/bin/activate
```

3. Install the developmental version of the project. This will include all the dependency necessary for developers. For details on the packages installed, see `pyproject.toml`.

Minimal example:

```bash
pip install -e .[dev]
```

With `uv`:
```bash
uv pip install -e .[dev]
```

4. Install pre-commit hooks to run all the checks before each commit.

```bash
pre-commit install
```

## Contributing to code

This is a very generic workflow.

1. Comment on an existing issue or open a new issue referencing your addition.

> [!TIP]
> Review and discussion on new code can begin well before the work is complete, and the more discussion the better!
> The development team may prefer a different path than you've outlined,
> so it's better to discuss it and get approval at the early stage of your work.


2. On your fork, create a new branch from main:

```bash
git checkout -b your_branch
```

3. Make the changes, lint, and format.

4. Commit your changes on this branch.

If you want to make sure all the tests will be run by github continuous integration,
make sure that your commit message contains `full_test`.

5. Run the tests locally; you can run spectfic tests to speed up the process:

```bash
pytest -v giga_connectome/tests/test_connectome.py::test_calculate_intranetwork_correlation
```

6. push your changes to your online fork. If this is the first commit, you might want to set up the remote tracking:

```bash
git push origin HEAD --set-upstream
```
In the future you can simply do:

```bash
git push
```
7. Submit a pull request from your fork of the repository.

8. Check that all continuous integration tests pass.

## Contributing to documentation

The workflow is the same as code contributions, with some minor differences.

1. Install the `[doc]` dependencies.

```bash
pip install -e '.[doc]'
```

2. After making changes, build the docs locally:

```bash
cd docs
make html
```

3. Submit your changes.

## Writing a PR

When opening a pull request, please use one of the following prefixes:

- **[ENH]** for enhancements
- **[FIX]** for bug fixes
- **[TEST]** for new or updated tests
- **[DOCS]** for new or updated documentation
- **[STYL]** for stylistic changes
- **[MAINT]** for refactoring existing code, any maintainace related things

Pull requests should be submitted early and often (please don't mix too many unrelated changes within one PR)!
If your pull request is not yet ready to be merged, please submit as a drafted PR.
This tells the development team that your pull request is a "work-in-progress", and that you plan to continue working on it.

One your PR is ready a member of the development team will review your changes to confirm that they can be merged into the main codebase.

### Running the demo

<!-- You can run a demo of the bids app by downloading some test data and atlases.

Run the following from the root of the repository.

```bash
pip install tox
tox -e test_data
python ./tools/download_templates.py
``` -->

You can run a demo with the test data shipped with the repo.
<!-- we need to update this instruction once the data is uploaded else where. -->
```bash
halfpipe2bids halfpipe2bids/tests/data/dataset-ds000030_halfpipe1.2.3dev outputs group
```

## Prepare a release

Currently this project is not pushed to PyPi.
We simply tag the version on the repository so users can reference the version of installation.
The release process will trigger a new tagged docker build of the software.

Switch to a new branch locally:

```bash
git checkout -b REL-x.y.z
```
First we need to prepare the release by updating the file `CHANGELOG.md` to make sure all the new features, enhancements, and bug fixes are included in their respective sections.

Finally, we need to change the title from x.y.z.dev to x.y.z

```markdown
## x.y.z

**Released MONTH YEAR**

### New
...
```
Add these changes and submit a PR:

```bash
git add docs/
git commit -m "REL x.y.z"
git push upstream REL-x.y.z
```

Once the PR has been reviewed and merged, pull from master and tag the merge commit:

```bash
git checkout main
git pull upstream main
git tag x.y.z
git push upstream --tags
```

## Post-release

At this point, the release has been made.

We also need to create a new section in `CHANGELOG.md` with a title and the usual New, Enhancements, Bug Fixes, and Changes sections for the version currently under development:

```markdown

## x.y.z+1.dev

**Released MONTH YEAR**

### New

### Fixes

### Enhancements

### Changes
```
