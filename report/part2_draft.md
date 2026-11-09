# Part 2: Word Embeddings and Semantic Drift

## Introduction

The goal of Part 2 is to study how the context and meaning of words change over time in news articles. We used two collections from the New York Times: P1, containing articles from 2005 to 2014, and P2, containing articles from 2015 to 2025, with partial coverage of 2025.

We trained three Word2Vec models to compare the two periods. Word2Vec represents each word as a numerical vector. Words that appear in similar contexts tend to have similar vectors, which allows us to examine relationships between words.

Our first model, M1, was trained on the older articles. Our second model, M2, started from M1 and continued learning from the newer articles. Our third model, M3, was trained independently on the newer articles. We also used a pretrained Google News model as an additional reference.

By comparing these models, we wanted to identify words whose associations changed over time and words whose meanings remained relatively stable.

## 2.1 Training the First Word2Vec Model (M1)

We first trained a Word2Vec model using the P1 corpus, which contains 563,736 articles and approximately 47.5 million tokens.

We used Gensim's Word2Vec implementation with the skip-gram method. In simple terms, skip-gram learns word representations by using a word to predict the words that appear around it. This helps the model learn which words are commonly used in similar contexts.

We used the following training settings:

| Parameter | Value |
|---|---|
| Vector size | 300 |
| Context window | 5 |
| Training method | Skip-gram |
| Negative samples | 5 |
| Minimum word frequency | 10 |
| Subsampling threshold | 0.001 |
| Training epochs | 5 |
| Random seed | 42 |
| Workers | 12 |

The model was trained on a Windows 11 Surface Pro with a 12-core Snapdragon X processor and approximately 15.6 GiB of usable RAM.

Training took **609.03 seconds, or approximately 10 minutes and 9 seconds**. The resulting model contained **66,853 words** in its vocabulary.

To check whether the model learned reasonable word relationships, we examined the eight nearest neighbors of five words.

| Word | Eight nearest neighbors |
|---|---|
| government | goverment, governments, clampdown, yingluck, rebuking, qurei, functionaries, yoshihiko |
| economy | recession, economic, weakening, economies, inflation, eurozone, growth, unemployment |
| baseball | league, selig, chass, leagues, sluggers, yankees, solarte, leaguers |
| music | edm, classical, electronica, rdio, songs, jazz, mog, tiësto |
| technology | technologies, coders, tech, virtualization, metron, computing, software, wimax |

The results show that the model learned recognizable relationships. For example, `economy` is close to `recession`, `inflation`, and `unemployment`, while `technology` is close to `computing` and `software`.

Not every neighbor is equally useful. Some are names, uncommon terms, or misspellings, such as `goverment`. This is expected because the model learns from the actual words and contexts in the corpus rather than from a dictionary.

Overall, these examples suggest that M1 learned useful word associations from the older news articles.

## 2.2 Continuing M1 on Newer Articles (M2)

Next, we wanted to see how the existing word representations would change when the model encountered newer articles.

Instead of starting over, we loaded M1 and continued training it on P2, which contains 406,984 articles and approximately 30.2 million tokens. We first updated the model's vocabulary and then trained it for five additional epochs.

The important difference is that **M2 begins with the word representations learned by M1**. It does not start with random vectors.

After continued training, the vocabulary increased from 66,853 to **71,819 words**, adding **4,966 words**. This process took **340.59 seconds, or approximately 5 minutes and 41 seconds**.

We saved M2 separately to preserve the original M1 model.

Continuing the training allows us to investigate how a word's representation changes when the model sees newer usage. Because M2 developed directly from M1, their vectors can be compared in the same coordinate system.

## 2.3 Training a Fresh Model on P2 (M3)

We also trained a separate Word2Vec model using only P2.

This model, M3, used the same training settings as M1, but it started from scratch rather than continuing an existing model.

M3 finished training in **356.78 seconds, or approximately 5 minutes and 57 seconds**, and contained **51,346 words**.

The difference between M2 and M3 is important:

- M2 learned from the older articles first and then adapted to the newer articles.
- M3 learned only from the newer articles, without knowledge of the earlier period.

This gives us two ways to study the newer language. We can examine how an older model adapts, or we can study what a model learns when it sees only newer articles.

However, M3 was trained independently, so its vector coordinates cannot be compared directly with M1. We address this problem using alignment in Section 2.5.

## 2.4 Using the Pretrained Google News Model

In addition to our three models, we used a pretrained Word2Vec model based on Google News. We refer to this model as GN.

The Google News model contains approximately **3 million word and phrase entries**, each represented by a 300-dimensional vector.

Unlike M1, M2, and M3, we did not train GN ourselves. We used it as an external reference to compare the word associations found in our news collections.

Because this model is much larger, we loaded and analyzed it on the Bilkent Leylak computing cluster rather than on the Surface Pro.

One important detail is that the Google News model is case-sensitive. For example, a lowercase word and its capitalized version may represent different meanings. To handle this consistently, our lookup procedure checked several capitalization forms and used the first available match.

We also handled phrases written with underscores, such as `New_York`, without automatically combining separate words into new phrases.

These rules allowed us to compare the models consistently, although differences in capitalization and word meaning could still affect the results.

We use GN more extensively in Section 2.6, where we compare its nearest neighbors with those from M1 and M2.
