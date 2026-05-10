<p align="center">
  <img src="assets/Tangible.png" style="width: 60%; height: auto;">
</p>

# Halgorightem
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
> Detecting AI Hallucinations **Before Them Happening**

## Whats Halgortihem
Halgorithem is a Custom Designed Algorithem For Detecting AI Hallucinations without Little to **Any AI Present in the Algo Itself**. Halgorithem was designed with speed in mind to quickly detect AI Hallucinating

## How does Halgorithem Work
Halgorithem works by Parsing your files and input into a Tree which is compared with file chunks which were made into trees. If something doesn't make sense, Halgorithem Flags it.

![How It Works](./assets/HowItWorks.png)
## Key Features

- **🔗 Fits Into Any AI workflow where responses are gened** <br>
Halgorithem can be integrated into AI Pipelines designed in python like LangGraph, CrewAI, PydanticAI and Microsoft AutoGen



## Screenshots

![Halgo Plugins](./assets/Halgorithem.gif)

## Installation

To run Halgorithem, follow these steps:

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```
   source venv/bin/activate
   ```

3. **Install the required modules:**
   ```
   pip install -r requirements.txt
   ```

4. **Download the spaCy English model (if it is not installed automatically):**
   ```
   python -m spacy download en_core_web_sm
   ```

5. **Run the benchmark:**
   ```
   python bench.py
   ```