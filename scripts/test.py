import os
import nltk

nltk.download('punkt_tab', download_dir='./nltk_data')
nltk.data.path.append(os.path.abspath('./nltk_data'))

from nltk.tokenize import sent_tokenize

print("✔ NLTK punkt setup complete.")
sample = "This is a test. NLTK should split this sentence."
print(sent_tokenize(sample))