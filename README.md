# Extractive-Summarizer-
Summarizes a given news article (via url) by extracting the N most "important" sentences (ongoing project, updates and new commits will be added in the next few weeks). Sentences are chosen by their cosine similarity to other sentences (after cleaning the text). 

Pre-Requisites:
gloVe 
newspaper (newspaper article extractor)

(ONGOING) Features / Optimizations To Implement:
- Improving data cleaning (removing links, quotations, halting lemmatization of proper nouns / entities)
- Updating the way a sentence is scored
  - Score is currently based on:
      - Cosine similarity with other sentences outside of the paragraph
      - Cosine similarity with sentences within the paragraph 
      - (1 = yes, 0 = no) Covers alternate sentence information above threshold (mean)
  - Added variables:
      - Cosine similarity with other sentences outside of the paragraph
      - Paragraph relevancy score, multiplied by cosine similarity with sentences w/i paragraph + information distribution
- Adding topic / entity extractor, extracting the major topics from the article + sentiment towards these nouns
- Creating a minimizing function to find optimal (and minimum) number of sentences to choose for summary 
      
