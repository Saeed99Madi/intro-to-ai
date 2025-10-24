# Contributing to Introduction to AI Course

Thank you for your interest in contributing to this course! We welcome contributions from students and educators.

## Ways to Contribute

### 1. Share Your Projects

Create a project based on the weekly material and share it with the community!

**Steps:**
1. Complete your project
2. Create a folder in the appropriate `community/` directory
3. Include:
   - Your code files
   - A README.md explaining your project
   - Screenshots or demo videos (optional)
   - Your name and contact info (optional)

**Example structure:**
```
1_search/community/your-name-project-name/
├── README.md
├── your_code.py
├── screenshots/
└── requirements.txt (if different from main course)
```

**README Template:**
```markdown
# Project Name

Brief description of what your project does.

## Author
Your Name (optional)

## What I Built
Explain what you built and why.

## What I Learned
Share your key learnings.

## How to Run
python your_code.py

## Screenshots
(Add images if applicable)

## Challenges
What was difficult? How did you overcome it?
```

### 2. Report Issues

Found a bug or typo? Please open an issue!

**Good issue includes:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your OS and Python version
- Error messages (if applicable)

### 3. Improve Documentation

Help make the course better by:
- Fixing typos or unclear explanations
- Adding more examples
- Translating content (contact maintainers first)
- Creating visual diagrams or illustrations

### 4. Add Code Examples

Enhance the learning experience by adding:
- Alternative implementations of algorithms
- Visualization improvements
- Additional exercises
- Real-world applications

### 5. Share Resources

Know of great resources? Add them to the relevant week's README or create a resources document.

## Contribution Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include docstrings for functions and classes

**Example:**
```python
def breadth_first_search(graph, start, goal):
    """
    Perform BFS to find a path from start to goal.

    Args:
        graph: Dictionary representing the graph
        start: Starting node
        goal: Target node

    Returns:
        List of nodes from start to goal, or None if no path exists
    """
    # Implementation here
    pass
```

### Notebook Guidelines

- Clear markdown explanations before code cells
- Run all cells before submitting
- Clear outputs if they contain large data
- Include visualizations where helpful
- Add learning objectives at the top

### Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Be specific ("Fix BFS algorithm bug" not "Fix bug")
- Reference issues if applicable ("Fixes #123")

**Good examples:**
- "Add visualization for A* search algorithm"
- "Fix typo in Week 1 Lab 2"
- "Improve explanation of heuristic functions"

### Testing

- Test your code before submitting
- Ensure it works with the course dependencies
- Include example usage

## Pull Request Process

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/intro-to-ai.git
   cd intro-to-ai
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Test thoroughly
   - Follow the style guidelines

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add clear description of changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes clearly
   - Link any related issues

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Community project
- [ ] Other (describe)

## Testing
How did you test your changes?

## Screenshots (if applicable)
Add screenshots or demo videos.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed my code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Tested on my machine
- [ ] No new warnings or errors
```

## Community Projects

### Requirements

Your community project should:
- Be based on course material
- Include a clear README
- Be well-documented
- Run without errors
- Not include plagiarized code

### Showcasing

Great projects may be:
- Featured in the main README
- Shared on social media
- Used as examples for future students

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers warmly
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy toward others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Plagiarism or academic dishonesty
- Sharing others' private information
- Spam or self-promotion

### Enforcement

Violations may result in:
1. Warning from maintainers
2. Temporary ban from contributing
3. Permanent ban from the community

Report issues to: [your.email@example.com]

## Recognition

Contributors will be acknowledged in:
- Contributors section of README
- Release notes (for significant contributions)
- Special thanks in weekly modules (for community projects)

## Questions?

- Check the [FAQ](docs/FAQ.md)
- Open an issue for course-related questions
- Email [your.email@example.com] for other inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make this course better for everyone!** 🙏

Your contributions help students around the world learn AI. Whether you're fixing a typo or building an amazing project, every contribution matters!

Happy learning and contributing! 🚀
