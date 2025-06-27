Contributing to the Meridian Retail AI Demo
First off, thank you for considering contributing! We welcome all contributions, from bug reports to new features. This document provides guidelines to help you get started.

Getting Started
To ensure you have a smooth development experience, please follow these steps to set up your environment:

Fork & Clone: Fork the repository to your own GitHub account and then clone it to your local machine.

Set Up Environment: The project uses a Makefile to simplify setup. From the root of the project, run:

make install

This command will create a Python virtual environment and install all the necessary dependencies listed in pyproject.toml.

Configure Environment: Copy the .env.example file to a new file named .env and fill in the required API keys and configuration variables.

Development Workflow
Create a Feature Branch: All new work should be done on a feature branch. Name your branch descriptively.

git checkout -b feature/your-amazing-feature

Make Small, Logical Commits: Try to keep your commits small and focused on a single logical change. This makes code reviews easier and helps maintain a clean project history. Write clear and concise commit messages.

Code Standards
To maintain a consistent and high-quality codebase, we use automated tools for linting and formatting.

Linting: We use ruff to identify potential issues.

Formatting: We use black for consistent code formatting.

Before you commit your changes, please run the following commands to ensure your code adheres to our standards:

# Check for linting issues and formatting errors
make lint

# Automatically fix linting and formatting issues
make format

Testing
We aim for a high level of test coverage. All new features or bug fixes must be accompanied by corresponding tests.

Before submitting your contribution for review, please run the full test suite to ensure that your changes haven't broken any existing functionality.

# Run the entire unit and integration test suite
make test

Pull Request Process
When you are ready to submit your contribution, please follow these steps:

Push your feature branch to your fork on GitHub.

Create a Pull Request (PR) to the main branch of the original repository.

Ensure your PR includes a descriptive title and a clear summary of the changes you've made. If it resolves an existing issue, please reference it (e.g., "Closes #123").

Pull Request Checklist
Before you submit your PR, please make sure you have completed the following:

[ ] My code follows the style guidelines of this project (make format).

[ ] I have run the linter and there are no new warnings (make lint).

[ ] I have added tests that prove my fix is effective or that my feature works.

[ ] All new and existing tests pass locally with my changes (make test).

[ ] I have written a clear and descriptive PR title and summary.

Thank you again for your contribution!