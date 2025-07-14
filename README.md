# Sentiment_Analysis
"This Python GUI app performs fast sentiment analysis on CSV data. Built with Tkinter, it features a sleek purple and jett-black theme. Leveraging NLTK's VADER, it processes thousands of records swiftly, providing confidence-based feedback and overall analysis statistics. Perfect for quick, visually intuitive sentiment insights."
-----
### How to Run This Sentiment Analysis App: A 3-Step Guide
This guide will help you get the Sentiment Analysis Application up and running on your local machine.
**Prerequisites:**

  * **Python 3.x:** Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
-----

**Step 1: Set Up Your Environment**

First, you need to install the necessary Python libraries.

1.  **Open your terminal or command prompt.**
2.  **Navigate to the project directory** where you saved (or cloned) the `sentiment.py` file. For example, if it's in `D:\Sentiment Analysis`, you would type:
    ```bash
    cd "D:\Sentiment Analysis"
    ```
3.  **Install the required libraries** by running this command:
    ```bash
    pip install pandas nltk
    ```
    Wait for the installation process to complete.

-----

**Step 2: Download NLTK Data**

The application uses NLTK's VADER lexicon for sentiment analysis. This data needs to be downloaded once.

1.  **In the same terminal or command prompt** (from Step 1), run the following command:
    ```bash
    python -c "import nltk; nltk.download('vader_lexicon')"
    ```
    This command will initiate the download. You should see messages indicating the download progress and confirmation upon completion.

-----

**Step 3: Run the Application**

Now you're ready to launch the sentiment analysis app\!

1.  **In the same terminal or command prompt** (from Step 2), execute the Python script:
    ```bash
    python sentiment.py
    ```
    The Sentiment Analysis Application GUI window should now appear on your screen. You can then click "Upload CSV" to select a dataset and begin analyzing.

-----
