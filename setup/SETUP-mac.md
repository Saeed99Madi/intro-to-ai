# Setup Guide for Mac

This guide will help you set up the Introduction to AI course on macOS.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Internet connection
- Admin access to your Mac

## Step 1: Install Homebrew (if not already installed)

Homebrew is a package manager for macOS that makes installing software easier.

Open Terminal (⌘ + Space, then type "Terminal") and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. After installation, you may need to add Homebrew to your PATH:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## Step 2: Install Python 3.10 or Later

Check if you have Python 3.10+:

```bash
python3 --version
```

If you don't have Python 3.10 or later, install it with Homebrew:

```bash
brew install python@3.10
```

Verify the installation:

```bash
python3 --version
```

You should see `Python 3.10.x` or later.

## Step 3: Install Git

Check if Git is installed:

```bash
git --version
```

If not installed:

```bash
brew install git
```

## Step 4: Clone the Repository

Navigate to where you want to store the course materials:

```bash
cd ~/Desktop  # or wherever you prefer
git clone <your-repository-url>
cd intro-to-ai
```

If you downloaded the ZIP file instead, extract it and navigate to the folder.

## Step 5: Create a Virtual Environment

Virtual environments keep your project dependencies isolated:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

**Important**: You need to activate the virtual environment every time you open a new terminal session to work on this course.

## Step 6: Install Dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -e .
```

This will install all required packages. It may take 5-10 minutes depending on your internet connection.

### Install NLTK Data (for Week 7)

```bash
python -c "import nltk; nltk.download('popular')"
```

### Install spaCy Model (for Week 7)

```bash
python -m spacy download en_core_web_sm
```

## Step 7: Verify Installation

Test that everything is working:

```bash
python -c "import numpy, scipy, sklearn, tensorflow, gradio; print('✓ All packages installed successfully!')"
```

If you see the success message, you're all set!

## Step 8: Launch Jupyter Lab

Start Jupyter Lab to access the course notebooks:

```bash
jupyter lab
```

This will open Jupyter Lab in your default web browser. If it doesn't open automatically, copy and paste the URL shown in the terminal.

## Step 9: Test a Notebook

1. In Jupyter Lab, navigate to `1_search/1_lab1.ipynb`
2. Try running the first few cells by pressing `Shift + Enter`
3. If the code runs without errors, you're ready to go!

## Optional: Install a Better Terminal

Consider installing iTerm2 for a better terminal experience:

```bash
brew install --cask iterm2
```

## Optional: Install VS Code

Visual Studio Code is a great editor for Python:

```bash
brew install --cask visual-studio-code
```

After installation, install the Python extension from within VS Code.

## Troubleshooting

### Issue: "command not found: python3"

**Solution**: Install Python using Homebrew (see Step 2)

### Issue: "command not found: pip"

**Solution**: Make sure your virtual environment is activated

```bash
source .venv/bin/activate
```

### Issue: TensorFlow installation fails

**Solution**: If you have an M1/M2/M3 Mac (Apple Silicon), you might need to install TensorFlow differently:

```bash
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration
```

### Issue: Jupyter Lab won't start

**Solution**: Make sure you're in the correct directory and your virtual environment is activated

```bash
cd ~/Desktop/intro-to-ai  # or your course directory
source .venv/bin/activate
jupyter lab
```

### Issue: Import errors when running notebooks

**Solution**: Make sure you're using the correct kernel in Jupyter Lab:
1. Click on the kernel name in the top-right corner
2. Select "Python 3 (.venv)" or the kernel associated with your virtual environment

### Issue: Permission denied errors

**Solution**: You may need to use `sudo` for some commands, but avoid this for pip installations

### Issue: Matplotlib plots not showing

**Solution**: Make sure you have the `%matplotlib inline` magic command in your notebook

## Updating the Course

To get the latest course materials:

```bash
cd ~/Desktop/intro-to-ai  # or your course directory
git pull origin main
```

Then update dependencies:

```bash
source .venv/bin/activate
pip install -e . --upgrade
```

## Uninstalling

If you need to remove the course:

1. Deactivate the virtual environment: `deactivate`
2. Delete the course folder: `rm -rf ~/Desktop/intro-to-ai`

## Next Steps

1. Read the main [README.md](../README.md) for course overview
2. Start with [Week 1: Search](../1_search/README.md)
3. Join the community and share your progress!

## Getting Help

- Check the [FAQ](../docs/FAQ.md)
- Review the [Troubleshooting Notebook](troubleshooting.ipynb)
- Open an issue on GitHub
- Search for error messages on Stack Overflow

## Useful Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Deactivate virtual environment
deactivate

# Start Jupyter Lab
jupyter lab

# Run a Gradio app
python 1_search/pathfinding_app.py

# Update course materials
git pull origin main

# Update packages
pip install -e . --upgrade
```

---

**All set?** Head to [Week 1: Search](../1_search/README.md) and start learning! 🚀

**Questions?** Open an issue on GitHub or check the FAQ.
