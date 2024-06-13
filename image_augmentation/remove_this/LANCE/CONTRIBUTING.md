
Setup instructions

1. Clone LANCE repo
```
git clone --recurse-submodules git@github.com:virajprabhu/LANCE.git
```
2. Create conda environment
```
cd lance; conda create -n lance python=3.8.5
```
3. Install requirements
```
pip3 install -r requirements.txt
```

# Contributing to LANCE

We welcome contributions of all kinds: code, documentation, feedback and support. If
 you use LANCE in your work (blogs posts, research, company) and find it
  useful, spread the word!  
  
This contribution borrows from and is heavily inspired by [Huggingface transformers](https://github.com/huggingface/transformers). 

If you encounter any steps that are missing from this guide, we'd appreciate if you open an issue or a pull request to improve it.

## How to contribute

There are 4 ways you can contribute:
* Issues: raising bugs, suggesting new features
* Fixes: resolving outstanding bugs
* Features: contributing new features
* Documentation: contributing documentation or examples

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. 

**However, we actively encourage that you** 
* file an incomplete issue than no issue at all
* suggest a feature that you are not sure how to implement, and even if you're unsure if it's a good idea

If you are unsure about something, don't hesitate to ask.

### Bugs

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues).

If you didn't find anything, please use the bug issue template to file a GitHub issue.  


### Features

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Contributing (Pull Requests)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`lance`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/virajprabhu/lance) by
   clicking on the 'Fork' button on the repository's page. 
   This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/lance.git
   $ cd lance
   $ git remote add upstream https://github.com/virajprabhu/lance.git
   ```

3. Create a new branch off of the `main` branch to hold your development changes:

   ```bash
   $ git fetch upstream
   $ git checkout -b a-descriptive-name-for-my-changes upstream/dev
   ```
   **Do not** work directly on the `main` branch.

4. LANCE manages dependencies using [`setuptools`](https://packaging.python.org/guides/distributing-packages-using-setuptools/). From the base of the `lance` repo, install the project in editable mode with all the extra dependencies with 

   ```bash
   $ pip install -e ".[all]"
   ```
   If LANCE was already installed in the virtual environment, remove it with `pip uninstall lance` before reinstalling it in editable mode with the above command.


5. Develop features on your branch.

   LANCE relies on `black` to format its source code
   consistently. After you make changes, autoformat them with:

   ```bash
   $ make autoformat
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/dev
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is done**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review. 

   **Important**:  Ensure that the you create a pull request onto lance's `master` branch. The drop down menus at the top of the "Open a pull request page" should look
   >base repository: **virajprabhu/lance**  base: **main** <- head repository: **\<your GitHub handle\>/lance**  compare: **\<your branch name\>** 
   

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`. These
   are useful to avoid duplicated work, and to differentiate it from PRs ready
   to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests.