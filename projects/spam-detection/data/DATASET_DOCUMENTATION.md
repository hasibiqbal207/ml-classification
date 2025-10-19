# Dataset Documentation

This document provides comprehensive information about the datasets used in the ML Classification project.

## Overview

The project includes three different types of classification problems, each with its own dataset:

| Problem Type | Dataset | Text Domain | Status |
|--------------|---------|-------------|---------|
| Binary | SMS Spam | SMS messages | ✅ Ready |
| Multiclass | 20 Newsgroups | News articles | ✅ Ready |
| Multilabel | GoEmotions | Reddit comments | ✅ Ready |

---

## 1. Binary Classification: SMS Spam Dataset

### Dataset Information
- **Source**: SMS Spam Collection v.1
- **Problem Type**: Binary Classification (Ham vs Spam)
- **Text Domain**: SMS messages
- **Language**: English
- **Encoding**: Latin-1

### Dataset Statistics
- **Total Messages**: 5,572
- **Ham Messages**: 4,825 (86.6%)
- **Spam Messages**: 747 (13.4%)
- **Class Imbalance**: Moderate (6.5:1 ratio)

### File Structure
```
projects/binary/data/
├── raw/
│   └── spam.csv                    # Original dataset
├── processed/
│   ├── train.csv                   # Training set (3,898 messages)
│   ├── val.csv                     # Validation set (835 messages)
│   ├── test.csv                    # Test set (837 messages)
│   ├── vocabulary.pkl              # Vocabulary (pickle format)
│   └── vocabulary.txt              # Vocabulary (text format)
└── external_datasets.txt           # Dataset source information
```

### Data Format
**Raw Data (spam.csv):**
- Column 1 (`v1`): Label (`ham` or `spam`)
- Column 2 (`v2`): SMS text content
- Columns 3-5: Empty columns

**Processed Data:**
- `label`: Binary label (`ham` or `spam`)
- `text`: Cleaned and normalized text

### Text Preprocessing
The SMS text undergoes the following preprocessing steps:
1. **Lowercase conversion**
2. **URL replacement**: URLs → `<url>`
3. **Email replacement**: Email addresses → `<email>`
4. **Phone replacement**: Phone numbers → `<phone>`
5. **Special character removal**: Non-alphanumeric characters removed
6. **Whitespace normalization**: Multiple spaces → single space

### Vocabulary
- **Size**: 4,222 words (minimum frequency = 2)
- **Special tokens**: `<PAD>` (0), `<UNK>` (1)
- **Most frequent words**: `i`, `to`, `you`, `a`, `the`, `u`, `and`, `in`

### Data Splits
- **Training**: 70% (3,898 messages)
- **Validation**: 15% (835 messages)
- **Test**: 15% (837 messages)
- **Stratification**: Maintains class balance across splits

### Sample Data
```
Label: ham
Text: "go until jurong point crazy available only in bugis n great world la e buffet cine there got amore wat"

Label: spam
Text: "free entry in 2 a wkly comp to win fa cup final tkts 21st may 2005 text fa to 87121 to receive entry question"
```

---

## 2. Multiclass Classification: 20 Newsgroups Dataset

### Dataset Information
- **Source**: 20 Newsgroups Dataset
- **Problem Type**: Multiclass Classification (20 classes)
- **Text Domain**: News articles
- **Language**: English
- **Encoding**: UTF-8

### Dataset Statistics
- **Total Categories**: 20 newsgroups
- **Total Documents**: ~20,000 news articles
- **Average Documents per Category**: ~1,000

### File Structure
```
projects/multi-class/data/
├── raw/
│   ├── alt.atheism.txt
│   ├── comp.graphics.txt
│   ├── comp.os.ms-windows.misc.txt
│   ├── comp.sys.ibm.pc.hardware.txt
│   ├── comp.sys.mac.hardware.txt
│   ├── comp.windows.x.txt
│   ├── misc.forsale.txt
│   ├── rec.autos.txt
│   ├── rec.motorcycles.txt
│   ├── rec.sport.baseball.txt
│   ├── rec.sport.hockey.txt
│   ├── sci.crypt.txt
│   ├── sci.electronics.txt
│   ├── sci.med.txt
│   ├── sci.space.txt
│   ├── soc.religion.christian.txt
│   ├── talk.politics.guns.txt
│   ├── talk.politics.mideast.txt
│   ├── talk.politics.misc.txt
│   ├── talk.religion.misc.txt
│   └── list.csv                    # Document mapping
└── external_datasets.txt           # Dataset source information
```

### Categories
The dataset includes 20 newsgroup categories organized into 6 main groups:

**Computer Technology:**
- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`

**Recreation:**
- `rec.autos`
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`

**Science:**
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `sci.space`

**Religion:**
- `alt.atheism`
- `soc.religion.christian`
- `talk.religion.misc`

**Politics:**
- `talk.politics.guns`
- `talk.politics.mideast`
- `talk.politics.misc`

**Miscellaneous:**
- `misc.forsale`

### Data Format
**Text Files:**
- Each file contains news articles from one category
- Articles include headers (From, Subject, etc.) and body text
- Raw text format with minimal preprocessing

**Mapping File (list.csv):**
- `newsgroup`: Category name
- `document_id`: Unique document identifier

### Sample Data
```
From: mathew <mathew@mantis.co.uk>
Subject: Alt.Atheism FAQ: Atheist Resources

Archive-name: atheism/resources
Alt-atheism-archive-name: resources
Last-modified: 24 Aug 1993
Version: 1.2
...
```

---

## 3. Multilabel Classification: GoEmotions Dataset

### Dataset Information
- **Source**: GoEmotions Dataset (Google Research)
- **Problem Type**: Multilabel Classification (27 emotion labels)
- **Text Domain**: Reddit comments
- **Language**: English
- **Encoding**: UTF-8

### Dataset Statistics
- **Total Comments**: 211,225
- **Total Labels**: 27 emotion categories
- **Average Labels per Comment**: ~1.2 (sparse multilabel)

### File Structure
```
projects/multi-label/data/
├── raw/
│   └── go_emotions_dataset.csv     # Original dataset
└── external_datasets.txt           # Dataset source information
```

### Data Format
**CSV Columns:**
- `id`: Unique comment identifier
- `text`: Reddit comment text
- `example_very_unclear`: Boolean flag for unclear examples
- 27 emotion columns (binary labels): `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`

### Emotion Categories
The dataset includes 27 emotion labels:

**Positive Emotions:**
- `admiration`, `amusement`, `approval`, `caring`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, `relief`

**Negative Emotions:**
- `anger`, `annoyance`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `fear`, `grief`, `nervousness`, `remorse`, `sadness`

**Neutral/Other:**
- `confusion`, `curiosity`, `desire`, `realization`, `surprise`, `neutral`

### Label Distribution
- **Most common**: `neutral` (~40% of comments)
- **Least common**: `grief`, `embarrassment` (~1% each)
- **Sparse labels**: Most comments have 0-2 emotion labels

### Sample Data
```
ID: eew5j0j
Text: "That game hurt."
Labels: sadness=1, others=0

ID: eeibobj
Text: "Man I love reddit."
Labels: joy=1, others=0

ID: ed2mah1
Text: "You do right, if you don't care then fuck 'em!"
Labels: neutral=1, others=0
```

---

## Data Processing Scripts

### Binary Classification
- **Script**: `projects/binary/scripts/preprocess_data.py`
- **Features**: Text cleaning, vocabulary creation, stratified splitting
- **Output**: Train/val/test splits + vocabulary

### Multiclass Classification
- **Status**: Processing script to be created
- **Requirements**: Text preprocessing, category mapping, train/val/test splits

### Multilabel Classification
- **Status**: Processing script to be created
- **Requirements**: Multilabel handling, sparse label processing, train/val/test splits

---

## Usage Guidelines

### Data Access
1. **Raw Data**: Located in `projects/{problem_type}/data/raw/`
2. **Processed Data**: Generated in `projects/{problem_type}/data/processed/`
3. **Scripts**: Located in `projects/{problem_type}/scripts/`

### Processing Commands
```bash
# Binary classification
cd projects/binary
python scripts/preprocess_data.py

# Multiclass classification (to be implemented)
cd projects/multi-class
python scripts/preprocess_data.py

# Multilabel classification (to be implemented)
cd projects/multi-label
python scripts/preprocess_data.py
```

### Data Loading
```python
import pandas as pd
import pickle

# Load processed data
train_data = pd.read_csv('data/processed/train.csv')
val_data = pd.read_csv('data/processed/val.csv')
test_data = pd.read_csv('data/processed/test.csv')

# Load vocabulary
with open('data/processed/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)
```

---

## Notes

- All datasets are ready for processing and model training
- Text preprocessing is tailored to each dataset's characteristics
- Data splits maintain class balance where applicable
- Vocabulary sizes are optimized for each problem type
- External dataset sources are documented in `external_datasets.txt` files

---

*Last updated: October 18, 2025*
