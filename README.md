# 🎬 Movie Review Text Analysis - NLP & Sentiment Classification (Python)

[![Python](https://img.shields.io/badge/Built%20With-Python-blue?logo=python)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-TextBlob-green)](https://textblob.readthedocs.io/)
[![WordCloud](https://img.shields.io/badge/Visualization-WordCloud-orange)](https://github.com/amueller/word_cloud)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📘 Project Overview

This project implements **comprehensive Natural Language Processing (NLP)** techniques to analyze movie review text data. The analysis includes **text preprocessing, word frequency analysis, sentiment classification, and data visualization** using industry-standard Python libraries.

**Key Features:**
- ✅ **Text Data Cleaning** with regex pattern matching and stopword removal
- ✅ **Word Frequency Analysis** identifying top 20 most common terms
- ✅ **WordCloud Visualization** showing word importance through size
- ✅ **Sentiment Analysis** using TextBlob polarity scoring
- ✅ **Sentiment Classification** into Positive, Negative, and Neutral categories
- ✅ **Professional Bar Graphs** for frequency distribution
- ✅ **Comprehensive Tutorial** for learning and replication

**Dataset:** 60 lines of professional movie reviews (8,271 characters) covering multiple films including detailed critiques and analysis.

**Analysis Type:** Exploratory Text Analysis with focus on sentiment distribution and keyword extraction.

---

## 🎯 Business Problem

Movie studios, streaming platforms, and film critics need to:
- Quickly **identify sentiment trends** across multiple reviews
- **Extract key themes** and frequently discussed topics
- **Visualize review content** for stakeholder presentations
- **Classify reviews** as positive, negative, or neutral for ratings
- **Understand critic perspectives** through word frequency analysis
- **Automate review processing** for large-scale analysis

---

## 📊 Dataset Description

**Source:** Movie review text file (Textdata.txt)

**Content Characteristics:**

| Attribute | Details |
|-----------|---------|
| **Total Lines** | 60 review segments |
| **Total Characters** | 8,271 characters |
| **Words (Raw)** | ~1,400 words |
| **Words (Cleaned)** | 727 words (after stopword removal) |
| **Unique Words** | 551 distinct terms |
| **Review Type** | Professional film criticism |
| **Movies Covered** | "From Hell" (primary) + additional film |
| **Writing Style** | Analytical, detailed, contextual |

**Sample Content:**
```
"films adapted from comic books have had plenty of success, whether 
they're about superheroes (batman, superman, spawn), or geared toward 
kids (casper) or the arthouse crowd (ghost world)..."
```

---

## 📈 Analysis Results

### **Word Frequency Distribution**

**Top 20 Most Common Words:**

| Rank | Word | Count | % of Total | Context |
|------|------|-------|------------|---------|
| 1 | film | 10 | 1.38% | Primary subject |
| 2 | like | 7 | 0.96% | Comparison/simile |
| 3 | dont | 7 | 0.96% | Negation |
| 4 | make | 7 | 0.96% | Creation/action |
| 5 | even | 6 | 0.83% | Emphasis |
| 6 | movie | 6 | 0.83% | Subject synonym |
| 7 | comic | 5 | 0.69% | Source material |
| 8 | get | 5 | 0.69% | Understanding |
| 9 | pretty | 5 | 0.69% | Qualifier |
| 10 | films | 4 | 0.55% | Plural form |
| 11 | world | 4 | 0.55% | Setting |
| 12 | really | 4 | 0.55% | Emphasis |
| 13 | say | 4 | 0.55% | Communication |
| 14 | little | 4 | 0.55% | Descriptor |
| 15 | good | 4 | 0.55% | Positive eval |
| 16 | see | 4 | 0.55% | Viewing |
| 17 | one | 4 | 0.55% | Quantifier |
| 18 | teen | 4 | 0.55% | Demographic |
| 19 | book | 3 | 0.41% | Source ref |
| 20 | moore | 3 | 0.41% | Author name |

**Key Observations:**
- Film terminology dominates (film + movie + films = 20 occurrences)
- Strong reference to comic book adaptations
- Mix of evaluative and descriptive language
- Specific references (Moore = Alan Moore, comic creator)

---

### **Sentiment Distribution**

**Overall Classification:**

```
╔════════════╦═══════╦════════════╦══════════════════╗
║ Sentiment  ║ Count ║ Percentage ║ Polarity Range   ║
╠════════════╬═══════╬════════════╬══════════════════╣
║ Positive   ║   23  ║   38.3%    ║ +0.051 to +0.250 ║
║ Negative   ║   14  ║   23.3%    ║ -0.750 to -0.051 ║
║ Neutral    ║   23  ║   38.3%    ║ -0.050 to +0.050 ║
╠════════════╬═══════╬════════════╬══════════════════╣
║ TOTAL      ║   60  ║   100.0%   ║ -1.0 to +1.0     ║
╚════════════╩═══════╩════════════╩══════════════════╝
```

**Sentiment Balance:**
- **Balanced Review:** Equal positive and neutral (38.3% each)
- **Moderate Criticism:** 23.3% negative suggests constructive critique
- **Professional Tone:** High neutral percentage indicates analytical writing

---

### **Top Positive Statements**

| Line | Polarity | Excerpt |
|------|----------|---------|
| 13 | +0.250 | "i don't think anyone needs to be briefed..." |
| 14 | +0.246 | "screenwriters do a good job of keeping him hidden..." |
| 1 | +0.175 | "films adapted from comic books have had plenty of success..." |

**Positive Indicators:** success, good, plenty

---

### **Top Negative Statements**

| Line | Polarity | Excerpt |
|------|----------|---------|
| 12 | -0.750 | "he befriends an unfortunate named mary kelly..." |
| 9 | -0.247 | "it's a filthy, sooty place where the whores..." |
| 3 | -0.130 | "to say moore and campbell thoroughly researched..." |

**Negative Indicators:** unfortunate, filthy, dismissive language

---

### **Neutral Statements**

| Line | Polarity | Excerpt |
|------|----------|---------|
| 8 | 0.000 | "the ghetto in question is, of course, whitechapel..." |
| 16 | 0.000 | "and from hell's ending had me whistling..." |
| 17 | 0.000 | "don't worry - it'll all make sense when you see it." |

**Characteristics:** Factual information, setting descriptions, temporal references

---

## 📂 Project Structure

```
TMNLP_MOVIE_REVIEW_TXT_ANALYSIS/
├── dashboards/              
├── data/
│   ├── raw/
│   │   └── Textdata.txt
│   └── processed/
├── environment/             
│   ├── environment.yml      
│   └── requirements.txt     
├── models/                  
├── notebooks/
│   └── 01_EDA_Modelling.ipynb 
├── reports/                 
│   ├── figures/
│   ├── sentiment_analysis.csv
│   ├── top_15_words_bargraph.png
│   └── wordcloud.png
├── src/                     
├── .gitignore              
├── LICENSE                 
├── main.py                 
└── README.md            
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-review-text-analysis.git
cd movie-review-text-analysis
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n text-analysis python=3.8
conda activate text-analysis
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
wordcloud>=1.8.0
textblob>=0.17.0
nltk>=3.6.0
seaborn>=0.11.0           # Optional: enhanced visualizations
```

---


---

## 📌 Important Notes

### ⚠️ Dataset Limitations

- **Small Corpus:** 60 lines may not represent full population
- **Source Bias:** Professional critics vs general audience
- **Temporal:** Reviews may be from specific time period
- **Sample Bias:** Primarily covers one film extensively

**For Production:**
- Expand dataset to 1,000+ reviews
- Include multiple sources (critics, audience, social media)
- Balance genres and time periods
- Consider rating correlations

---

### 🎯 Sentiment Accuracy

**Evaluation Considerations:**
- TextBlob accuracy: ~70-80% on general text
- Movie reviews may have lower accuracy
- Sarcasm and irony can mislead
- Context window limitations

**Validation Approach:**
- Manual review of random 10% sample
- Compare with human-annotated labels
- Test on held-out validation set
- Monitor false positive/negative rates

---

## 📖 References

### Academic Papers
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers.
- Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval, 2(1-2), 1-135.
- Mohammad, S. (2016). *Sentiment Analysis: Detecting Valence, Emotions, and Other Affectual States from Text*. Emotion Measurement, 201-237.

### Documentation & Libraries
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [NLTK Book](https://www.nltk.org/book/)
- [WordCloud GitHub](https://github.com/amueller/word_cloud)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Industry Resources
- [Google NLP Best Practices](https://cloud.google.com/natural-language/docs/basics)
- [AWS Comprehend](https://aws.amazon.com/comprehend/)
- [Towards Data Science - NLP](https://towardsdatascience.com/tagged/natural-language-processing)

---

## 👨‍💼 Author

**William C. Phiri**  
📧 [wphiri@beda.ie]  
🔗 [LinkedIn](https://www.linkedin.com/in/william-phiri-866b8443/)  
🐙 [GitHub: Kochezz](https://github.com/kochezz)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 William C. Phiri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- Dataset represents professional film criticism
- Built as part of Data Science portfolio development
- Inspired by NLP best practices and industry standards
- Thanks to the open-source community for excellent libraries
- Special appreciation for TextBlob and WordCloud maintainers

---


---

**⭐ If you found this project helpful, please consider giving it a star!**

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Additional sentiment analysis methods
- More visualization types
- Performance optimization
- Extended documentation
- Test coverage
- Bug fixes

---

## 📧 Contact & Support

**Questions?** Open an issue on GitHub

**Collaboration?** Email wphiri@beda.ie

**Feedback?** Pull requests welcome

---

**Last Updated:** November 12, 2025  
**Version:** 1.0.0  
**Python Version:** 3.8+

---

**End of Documentation**
