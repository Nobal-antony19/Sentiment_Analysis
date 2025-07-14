import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import threading
import time

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon... This may take a moment.")
    nltk.download('vader_lexicon')
    print("VADER lexicon downloaded.")

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis App")
        self.root.geometry("1000x750") # Increased window height for conclusion section
        self.root.minsize(800, 650) # Minimum size
        self.root.configure(bg="#E0BBE4") # Light, neutral muted purple background

        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()
        self.df = None # To store the loaded DataFrame

        self.create_widgets()

    def create_widgets(self):
        # Styling for ttk widgets
        style = ttk.Style()
        style.theme_use('clam') # Use 'clam' theme as a base for better customization

        # Configure frame style
        style.configure('TFrame', background='#E0BBE4') # Light, neutral muted purple
        style.configure('TLabel', background='#E0BBE4', foreground='black', font=('Arial', 12))

        # Configure Button style for "jett black" and slightly rounded appearance
        # Tkinter buttons don't have direct border-radius, but padding and relief='flat'
        # with no borderwidth can give a softer, slightly rounded feel.
        style.configure('TButton',
                        background='#1A1A1A', # Very dark grey, close to jett black
                        foreground='white',
                        font=('Arial', 12, 'bold'),
                        padding=[15, 8], # Increased horizontal padding for a softer look
                        relief='flat', # Flat relief for a modern, slightly rounded appearance
                        borderwidth=0) # No border for a smoother look
        style.map('TButton',
                   background=[('active', '#333333')], # Darker black on hover
                   foreground=[('active', 'white')])

        # Configure Treeview style
        style.configure("Treeview",
                        background="#FFFFFF", # White background for table
                        foreground="black",
                        rowheight=25,
                        fieldbackground="#FFFFFF",
                        font=('Arial', 10))
        style.map('Treeview', background=[('selected', '#957DAD')]) # Slightly darker muted purple on selection
        style.configure("Treeview.Heading",
                        font=('Arial', 11, 'bold'),
                        background="#1A1A1A", # Jett black heading
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading",
                  background=[('active', '#333333')])

        # Configure Progressbar style
        style.configure("TProgressbar",
                        troughcolor='#D3D3D3', # Light grey trough
                        background='#957DAD') # Slightly darker muted purple progress bar

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Sentiment Analysis Application", font=('Arial', 24, 'bold'), foreground='black', background='#E0BBE4')
        title_label.pack(pady=10)

        # File upload section
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(pady=10)

        self.file_label = ttk.Label(file_frame, text="No CSV file selected.", width=40, anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=10)

        upload_button = ttk.Button(file_frame, text="Upload CSV", command=self.upload_csv)
        upload_button.pack(side=tk.LEFT)

        # Analysis button
        self.analyze_button = ttk.Button(main_frame, text="Analyze Sentiment", command=self.start_analysis_thread, state=tk.DISABLED)
        self.analyze_button.pack(pady=15)

        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready.", font=('Arial', 10), foreground='black')
        self.status_label.pack(pady=5)

        # Results display (Treeview)
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.tree = ttk.Treeview(results_frame, columns=("Text", "Sentiment", "Confidence"), show="headings")
        self.tree.heading("Text", text="Original Text")
        self.tree.heading("Sentiment", text="Sentiment")
        self.tree.heading("Confidence", text="Feedback")

        # Set column widths (adjust as needed)
        self.tree.column("Text", width=400, anchor=tk.W)
        self.tree.column("Sentiment", width=100, anchor=tk.CENTER)
        self.tree.column("Confidence", width=150, anchor=tk.CENTER)

        # Add scrollbar to Treeview
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Tag configurations for sentiment colors (kept for clarity, ensure contrast)
        self.tree.tag_configure('positive', foreground='#2E7D32')  # Dark Green
        self.tree.tag_configure('negative', foreground='#C62828')  # Dark Red
        self.tree.tag_configure('neutral', foreground='#424242')   # Dark Grey

        # Tag configurations for feedback colors (kept for clarity, ensure contrast)
        self.tree.tag_configure('strong', background='#DCE775', foreground='#33691E') # Lime Green/Dark Green
        self.tree.tag_configure('moderate', background='#FFF176', foreground='#F57F17') # Yellow/Orange
        self.tree.tag_configure('ok_low', background='#EF9A9A', foreground='#B71C1C') # Red/Dark Red

        # Conclusion section
        conclusion_frame = ttk.Frame(main_frame, padding="10 0 10 0")
        conclusion_frame.pack(fill=tk.X, pady=10)

        self.total_records_label = ttk.Label(conclusion_frame, text="Total Records: N/A", font=('Arial', 12, 'bold'), foreground='black')
        self.total_records_label.pack(side=tk.LEFT, padx=10)

        self.accuracy_label = ttk.Label(conclusion_frame, text="Overall Accuracy (Strong Feedback): N/A", font=('Arial', 12, 'bold'), foreground='black')
        self.accuracy_label.pack(side=tk.RIGHT, padx=10)


    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_label.config(text=f"Selected: {file_path.split('/')[-1]}")
            self.status_label.config(text="File loaded, ready for analysis.")
            self.analyze_button.config(state=tk.NORMAL)
            self.tree.delete(*self.tree.get_children()) # Clear previous results
            self.total_records_label.config(text="Total Records: N/A") # Reset conclusion
            self.accuracy_label.config(text="Overall Accuracy (Strong Feedback): N/A") # Reset conclusion
            try:
                self.df = pd.read_csv(file_path)
                if self.df.empty:
                    messagebox.showwarning("Empty File", "The selected CSV file is empty.")
                    self.analyze_button.config(state=tk.DISABLED)
                    self.df = None
                    return
                # Assume the first column is the text to analyze.
                # In a real app, you might let the user select the column.
                if len(self.df.columns) == 0:
                    messagebox.showerror("Invalid CSV", "The CSV file has no columns. Please ensure it's properly formatted.")
                    self.analyze_button.config(state=tk.DISABLED)
                    self.df = None
                    return
                self.status_label.config(text=f"Loaded {len(self.df)} records. Click 'Analyze Sentiment'.")
            except Exception as e:
                messagebox.showerror("Error Reading CSV", f"Could not read CSV file: {e}")
                self.file_label.config(text="Error loading file.")
                self.analyze_button.config(state=tk.DISABLED)
                self.df = None
        else:
            self.file_label.config(text="No CSV file selected.")
            self.status_label.config(text="Ready.")
            self.analyze_button.config(state=tk.DISABLED)
            self.df = None
            self.total_records_label.config(text="Total Records: N/A")
            self.accuracy_label.config(text="Overall Accuracy (Strong Feedback): N/A")

    def start_analysis_thread(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("No Data", "Please upload a CSV file first.")
            return

        self.analyze_button.config(state=tk.DISABLED)
        self.progress_bar.config(value=0, mode="determinate")
        self.status_label.config(text="Analyzing sentiment... Please wait.")
        self.tree.delete(*self.tree.get_children()) # Clear previous results
        self.total_records_label.config(text="Total Records: N/A") # Reset conclusion
        self.accuracy_label.config(text="Overall Accuracy (Strong Feedback): N/A") # Reset conclusion


        # Start analysis in a separate thread to keep GUI responsive
        analysis_thread = threading.Thread(target=self.analyze_sentiment)
        analysis_thread.start()

    def analyze_sentiment(self):
        start_time = time.time()
        results = []
        total_records = len(self.df)
        strong_feedback_count = 0 # Counter for strong feedback
        text_column_name = self.df.columns[0] # Assuming first column is text

        for i, row in self.df.iterrows():
            text = str(row[text_column_name]) # Ensure text is string
            if pd.isna(text) or text.strip() == "":
                sentiment = "N/A"
                feedback = "N/A"
                sentiment_tag = ''
                feedback_tag = ''
            else:
                vs = self.analyzer.polarity_scores(text)
                compound_score = vs['compound']

                # Determine sentiment
                if compound_score >= 0.05:
                    sentiment = "Positive"
                    sentiment_tag = 'positive'
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                    sentiment_tag = 'negative'
                else:
                    sentiment = "Neutral"
                    sentiment_tag = 'neutral'

                # Determine feedback based on absolute compound score (confidence)
                abs_compound = abs(compound_score)
                if abs_compound >= 0.85:
                    feedback = "Strong (85%+)"
                    feedback_tag = 'strong'
                    strong_feedback_count += 1 # Increment count for strong feedback
                elif abs_compound >= 0.75:
                    feedback = "Moderate (75-85%)"
                    feedback_tag = 'moderate'
                else:
                    feedback = "Ok to Low (<75%)"
                    feedback_tag = 'ok_low'

            results.append((text, sentiment, feedback, sentiment_tag, feedback_tag))

            # Update progress bar
            progress = (i + 1) / total_records * 100
            self.root.after(1, self.progress_bar.config, {'value': progress}) # Update GUI from main thread

        end_time = time.time()
        duration = end_time - start_time

        # Update GUI elements from the main thread after analysis is complete
        self.root.after(1, self.display_results, results, duration, total_records, strong_feedback_count)

    def display_results(self, results, duration, total_records, strong_feedback_count):
        for text, sentiment, feedback, sentiment_tag, feedback_tag in results:
            self.tree.insert("", tk.END, values=(text, sentiment, feedback), tags=(sentiment_tag, feedback_tag))

        self.status_label.config(text=f"Analysis complete for {len(results)} records in {duration:.2f} seconds.")
        self.analyze_button.config(state=tk.NORMAL)
        self.progress_bar.config(value=100) # Ensure it shows 100% at the end

        # Update conclusion section
        self.total_records_label.config(text=f"Total Records: {total_records}")
        if total_records > 0:
            accuracy_percentage = (strong_feedback_count / total_records) * 100
            self.accuracy_label.config(text=f"Overall Accuracy (Strong Feedback): {strong_feedback_count} / {total_records} ({accuracy_percentage:.2f}%)")
        else:
            self.accuracy_label.config(text="Overall Accuracy (Strong Feedback): N/A")


        # Provide a final message box for very large datasets if it took too long
        if duration > 120: # Over 2 minutes
            messagebox.showinfo("Analysis Complete",
                                f"Analysis finished in {duration:.2f} seconds. For extremely large datasets, consider optimizing CSV reading or using more powerful hardware.")
        elif duration > 70: # Over 70 seconds for 5000+ records
            messagebox.showinfo("Analysis Complete",
                                f"Analysis finished in {duration:.2f} seconds. It was a large dataset, performance is within expected range.")
        else:
            messagebox.showinfo("Analysis Complete",
                                f"Analysis finished in {duration:.2f} seconds. Fast and smooth!")


if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
