# HOW TO CHECK THAT YOUR ASSIGNMENT CODE PASSES AUTOGRADING

To streamline assessing reproducibility, the NLP teaching team uses an autograding workflow to run a certain number of functions automatically.
However, due to GitHub restrictions, we can only afford to run this code once when you submit your assignment by merging `main->feedback`.

However, there is a way for you to run the autograding on your own to check that your code passes it, which is the object of this tutorial.

## Explaining the autograding code

The autograding workflow is a GitHub Action script contained in `.github/workflows/score.yml`, which:

1. Is triggered when a push happens on branch feedback, which in your case is when you will merge the PR `main->feedback` to submit your assignment.
2. Starts a "job" by creating a virtual machine running the ubuntu operating system on GitHub's servers.
3. Runs the following:
    - Looks for an environment.yml file containing an environment called `nlp-final` at the root of your repository, use it to setup conda, or returns an error if the file isn't there or the environment has a different name.
    - Runs `pylint` to check the code quality (did you add doctrings to your classes and functions? are there undefined variables, missing imports, or invalid method calls?...) and post a report in the PR.
    - Runs main.py to check that the code doesn't return an error.

## How to run autograding on your own

In your personnal assignment repository, modifying `.github/workflows/score.yml` would return an error due to a feature of GitHub Classroom meant to prevent students hacking their way to a good grade.
As such, to run the autograding on your own, you will need to:
1. Fork your asssignment repository.
2. Edit the `.github/workflows/score.yml` file in your forked repository (where it isn't protected by GitHub Classroom).

---

### Step 1/2: forking your assignment to create a copy in your personal Github account
* On the root page of your assignment repository on GitHub (the one hosted on the organisation UCL-ELEC0141-26), you should see in the top right corners a series of buttons, with one of them labeled `fork`.
* Clicking on the `fork` button will take you to a "Create a fork" webpage. 

> [!CAUTION]
> Make sure that the owner of the fork is your personnal GitHub account, not UCL-ELEC0141-26.

* Untick the "Copy the main branch only" box, and create the fork. You should be taken to a copy of the repository hosted on your account.

---

### Step 2/2: modifying the autograding workflow to be triggered anytime something is pushed on main.
* In your forked repository, click on the Action tab (in the top left corner of the page, next to Pull Request).
* If you are presented with a message saying "Worksflows aren't bein run in this repository", click on the "Enable actions in this repository" button.
* Now, open the `.github/workflows/score.yml` file in branch main.
* At the top of the code, replace
  ```yaml
  name: Scoring code quality

  on:
    push:
      branches:
        - feedback
    workflow_dispatch: 

  ```
* with
    ```yaml
  name: Scoring code quality

  on:
    push:
      branches:
        - main
    workflow_dispatch:

  ```
* Commit your changes, and go to the PR `your_fork.main->your_fork.feedback` (if no such PR exists, create one). Now everytime you commit something on branch main of your fork, you will see in the PR a workflow that will indicate the result of the autograding script.
  
## Warnings

> [!WARNING]
> We will only look at your assignment repository hosted on the UCL-ELEC0141-26 organisation, not at your forked repositories.
> As such, any changes to your code made in your fork will not be taken into account unless you also include it in your assignment repository.

> [!WARNING]
> If you make changes in your fork and decide to use a Pull request `your_GitHub/your_fork.main->UCL-ELEC0141-26/your_assignment.main` to update your assignment, make sure that you do not change the `.github/workflows/score.yml` when you merge.
