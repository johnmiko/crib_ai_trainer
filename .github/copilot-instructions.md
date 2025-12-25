Activate the virtual environment with .\.venv\Scripts\Activate.ps1 before doing any work
- If you ever write any code that involves calculations, make sure to write a unit test to test that the calculation is correct
- When you are done completing any task, run the projects test suite to confirm that none of the existing functionality is broken
Once you have written the test, run the test iteratively and fix things until it passes

If you are coding in python, use log statements instead of print statements (logger = getLogger(__name__))