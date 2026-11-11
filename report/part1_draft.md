# CS533 Homework 1: Part 1 draft

**Word Embeddings, Semantic Drift, and Retrieval Across Two Decades of News**

Student: **[name to be confirmed]**  
Course: **CS533**  
Email: **[email to be confirmed]**  
Assignment: **1**

This draft covers Sections 1.1-1.5. It separates results already recorded in the
notebook, results reproduced from saved corpus files, and reporting statistics
recomputed from official metadata. The official files were used to validate
independent reconstruction and to describe the test sets; they were not used to
choose the documents, queries, or relevance judgments in the reconstruction.

## 1.1 Dataset inspection and temporal coverage

The One Million News dataset contains New York Times articles published between
2005 and 2025. Each record has a URL, publication timestamp, article text,
`section_name`, and `mapped_section`. The text combines the headline, abstract,
and lead paragraph. The nine mapped categories provide broad editorial labels;
they are distinct from the 32 URL-defined retrieval topics used in Section 1.5.

The original HDF5 table has **1,007,004 rows and five columns**, as recorded in
the saved exploration notebook and reproduced in the consolidated review. I used the publication timestamp to derive each
article's year. The first period, P1, covers 2005-2014 inclusively; P2 covers
2015-2025 inclusively. These periods do not overlap, and 2025 is a partial year.

**Table 1. Dataset and period definitions.**

| Item | Value | Measurement source |
| --- | --- | --- |
| Original article count | 1,007,004 | Saved notebook, original HDF5 inspection |
| Original columns | 5 | Saved notebook, original HDF5 inspection |
| Broad mapped categories | 9 | Saved notebook category tables |
| P1 | 2005-2014 | Assignment period definition |
| P2 | 2015-2025 | Assignment period definition |
| Raw P1 articles | 595,708 | Saved notebook raw period/category table |
| Raw P2 articles | 411,296 | Saved notebook raw period/category table |

The year-by-category and period-by-category summaries below describe the
**original, uncleaned dataset**. They should not be confused with the smaller
cleaned training counts in Section 1.4.

**Table 2. Original article counts by publication year and mapped category.**

| Year | Business & Economy | Culture, Arts & Lifestyle | Health & Wellbeing | International News | Lifestyle & Leisure | Other | Politics & Government | Sports & Athletics | Tech, Science & Education | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2005 | 15,841 | 11,800 | 1,418 | 25,981 | 4,048 | 289 | 10,949 | 7,138 | 2,029 | 79,493 |
| 2006 | 26,969 | 16,321 | 1,868 | 32,215 | 4,999 | 854 | 14,688 | 11,946 | 3,602 | 113,462 |
| 2007 | 18,128 | 9,577 | 996 | 15,592 | 4,071 | 1,179 | 11,612 | 9,040 | 3,811 | 74,006 |
| 2008 | 13,245 | 6,510 | 1,054 | 9,707 | 3,332 | 706 | 10,292 | 7,721 | 2,880 | 55,447 |
| 2009 | 1,283 | 1,609 | 270 | 1,556 | 806 | 81 | 694 | 750 | 405 | 7,454 |
| 2010 | 5,333 | 6,060 | 828 | 6,440 | 1,661 | 260 | 3,122 | 3,783 | 1,501 | 28,988 |
| 2011 | 6,382 | 7,285 | 924 | 7,253 | 1,894 | 1,061 | 6,553 | 5,253 | 3,594 | 40,199 |
| 2012 | 11,431 | 12,955 | 1,409 | 14,210 | 3,246 | 2,068 | 12,924 | 10,485 | 7,481 | 76,209 |
| 2013 | 9,539 | 10,675 | 1,176 | 12,013 | 3,096 | 1,924 | 8,697 | 6,802 | 4,736 | 58,658 |
| 2014 | 11,216 | 11,724 | 1,126 | 12,862 | 3,953 | 1,544 | 9,778 | 5,565 | 4,024 | 61,792 |
| 2015 | 7,989 | 11,586 | 1,018 | 11,575 | 3,153 | 813 | 10,007 | 5,183 | 3,604 | 54,928 |
| 2016 | 6,472 | 10,067 | 1,325 | 9,596 | 2,567 | 655 | 8,192 | 4,529 | 3,379 | 46,782 |
| 2017 | 5,567 | 9,920 | 1,391 | 8,521 | 2,268 | 516 | 5,704 | 3,463 | 3,127 | 40,477 |
| 2018 | 4,711 | 9,688 | 1,688 | 7,859 | 2,368 | 572 | 6,180 | 3,082 | 2,730 | 38,878 |
| 2019 | 4,218 | 9,162 | 1,917 | 6,957 | 2,337 | 554 | 7,248 | 2,830 | 2,386 | 37,609 |
| 2020 | 4,840 | 7,672 | 2,366 | 6,573 | 2,179 | 541 | 9,389 | 2,427 | 3,526 | 39,513 |
| 2021 | 4,781 | 6,922 | 2,114 | 7,525 | 1,642 | 693 | 7,819 | 3,129 | 2,712 | 37,337 |
| 2022 | 4,399 | 6,497 | 1,921 | 7,636 | 1,664 | 967 | 6,286 | 2,820 | 2,391 | 34,581 |
| 2023 | 3,865 | 6,597 | 2,080 | 7,058 | 1,216 | 1,315 | 6,378 | 1,449 | 2,214 | 32,172 |
| 2024 | 3,797 | 7,203 | 2,641 | 8,024 | 1,249 | 1,567 | 8,783 | 87 | 2,400 | 35,751 |
| 2025 | 1,525 | 2,476 | 988 | 2,789 | 428 | 523 | 3,567 | 29 | 943 | 13,268 |
| Total | 171,531 | 182,306 | 30,518 | 221,942 | 52,177 | 18,682 | 168,862 | 97,511 | 63,475 | 1,007,004 |

These counts were reproduced from the original HDF5 in the consolidated run, before cleaning.

**Table 3. Original article counts by mapped category and period.**

| Mapped category | Raw P1 | Raw P2 |
| --- | ---: | ---: |
| Business & Economy | 119,367 | 52,164 |
| Culture, Arts & Lifestyle | 94,516 | 87,790 |
| Health & Wellbeing | 11,069 | 19,449 |
| International News | 137,829 | 84,113 |
| Lifestyle & Leisure | 31,106 | 21,071 |
| Other | 9,966 | 8,716 |
| Politics & Government | 89,309 | 79,553 |
| Sports & Athletics | 68,483 | 29,028 |
| Tech, Science & Education | 34,063 | 29,412 |
| Total | 595,708 | 411,296 |

### Coverage observations

The monthly counts reveal a substantial coverage gap in 2009: January through
October have no records, whereas November has **3,752** and December **3,702**.
Thus the low annual total of **7,454** does not represent a fully observed year.
The saved notebook's monthly counts support an interpretation of incomplete
dataset coverage; they do not establish why those records are absent.

The 2025 data are also incomplete. The notebook records **3,237**, **2,937**,
**3,224**, and **3,177** articles in January-April, followed by **693** in May and
none in June-December. The annual total is **13,268**. Comparisons involving 2025
must therefore account for partial-year exposure instead of treating its count
as a full-year publication rate. The freshly measured last publication
timestamp is `2025-05-07T07:59:42+00:00`, consistent with partial coverage in May.

Sports coverage declines sharply in the later data. The raw `Sports & Athletics`
category contains **1,449** articles in 2023, **87** in 2024, and **29** in partial
2025. The monthly transition is especially clear in 2023: September contains
**97** sports articles, followed by **14**, **8**, and **7** in October-December.
The drop may reflect changes in the dataset's collection or category coverage,
editorial classification, or publication practices. The counts alone cannot
identify a cause and do not demonstrate a change in public interest in sports.
These observations were recorded in the saved notebook and reproduced from
the original HDF5 during the consolidated review. They are descriptive coverage
checks, not a causal analysis.

## 1.2 Exact cleaning procedure

I applied the three cleaning rules in the order specified in Part 1.2 and
Appendix A.1:

1. Remove articles whose text contains fewer than 10 whitespace-separated
   tokens. The implementation counts `text.fillna("").str.count(r"\S+")`;
   this step does not call the later preprocessing function.
2. Remove duplicate article text, keeping the first surviving occurrence in
   input order.
3. Remove duplicate `web_url` values, again keeping the first surviving
   occurrence.

The order is significant: duplicate detection is applied to the rows surviving
the preceding step. Exact text equality is used; this is not approximate
near-duplicate detection. I retained the original DataFrame while constructing
the cleaned result, so the raw-data inspection and cleaning counts remained
separate.

**Table 4. Cleaning removals and remaining article counts.**

| Stage | Removed at this stage | Remaining |
| --- | ---: | ---: |
| Original dataset | - | 1,007,004 |
| Fewer than 10 whitespace-separated tokens | 28,799 | 978,205 |
| Duplicate article text | 6,914 | 971,291 |
| Duplicate URL | 571 | **970,720** |

These counts were recorded in the notebook and independently reproduced by the
existing command-line validation workflows. The cleaned total matches the
assignment's checksum. Matching the checksum is a useful check, but the later
record-by-record comparisons provide a stronger test than aggregate counts
alone.

## 1.3 Preprocessing decisions and token statistics

The reusable pipeline is `preprocess_text()` in `src/preprocessing.py`. It
returns a list of tokens and is reused by the token-counting and corpus-writing
functions. Its decisions are as follows:

- **Lowercasing:** merge case variants such as `News` and `news`. This also loses
  potentially useful distinctions, including an acronym `US` versus the word
  `us`.
- **Apostrophes:** normalize left/right curly apostrophes, the modifier-letter
  apostrophe, and the full-width apostrophe to ASCII `'`.
- **Dotted acronyms:** collapse sequences such as `U.S.`, `N.B.A.`, `F.B.I.`, and
  `E.P.A.` to `us`, `nba`, `fbi`, and `epa` before splitting punctuation. A final
  period is optional. This is a spelling rule rather than semantic abbreviation
  recognition, so it also collapses forms such as `e.g.` to `eg`.
- **Possessives:** remove singular `'s` and plural `s'` endings. The rule cannot
  distinguish a possessive from an `'s` contraction, so `it's` becomes `it`.
  Other internal apostrophes remain, for example in `don't`.
- **Punctuation and hyphens:** treat remaining punctuation, hyphens, and
  underscores as token boundaries. Thus `state-of-the-art` becomes four tokens.
- **Numbers:** preserve numeric tokens, including years. Because punctuation
  splits tokens, `3.14` becomes `3`, `14`, and `1,000` becomes `1`, `000`; no
  numeric interpretation is attempted.
- **Stop words:** keep them for Word2Vec training. No stop-word removal is
  applied to the token statistics or saved corpora.
- **Missing text:** empty strings and non-string missing values produce no
  tokens. There is no stemming or lemmatization. Unicode letters and digits
  are retained, but there is no accent normalization or language-specific
  segmentation; decomposed combining accents can split words.

The following illustrative examples were run through the current reusable
function for this draft. They describe the implemented pipeline, rather than
the older exploratory regex shown in some notebook cells.

**Example 1. Case, acronym, possessive, and number handling.**

```text
Before: The U.S. President’s NEW policy in 2025!
After:  ['the', 'us', 'president', 'new', 'policy', 'in', '2025']
```

**Example 2. Acronym, possessive, hyphen, and stop-word handling.**

```text
Before: The N.B.A. team's state-of-the-art pre-season plan for 2025.
After:  ['the', 'nba', 'team', 'state', 'of', 'the', 'art', 'pre',
         'season', 'plan', 'for', '2025']
```

The acronym rule addresses a concrete document/query mismatch: `U.S.` and the
query word `us` both become `us`; similarly `N.B.A.`/`nba`, `F.B.I.`/`fbi`, and
`E.P.A.`/`epa` obtain the same token spelling. The notebook also records an
actual article excerpt beginning `U.S. Vows Big Increase in Aid...`, which the
reusable function maps to tokens beginning `us`, `vows`, `big`, `increase`.
These checks establish the normalization behavior; they do not measure a
retrieval-effectiveness improvement.

The token-counting function processes one article at a time. Repeated tokens
and stop words contribute to the total, while a separate vocabulary set for
each period provides the distinct-token count. Memory therefore depends on the
two vocabularies and the current article's tokens, not on a stored list of every
article's token list. Token statistics are reported with the training corpora
in Table 5.

## 1.4 Time-split training corpora

I used all cleaned articles in each non-overlapping period, without sampling.
`write_training_corpora()` calls the same `preprocess_text()` function and
writes each article immediately as space-separated tokens followed by a
newline. Relative article order within a period is preserved.

The two UTF-8 files are `corpus_P1.txt` and `corpus_P2.txt`, stored externally
under `../data/corpora/`, outside the Git repository. The writer refuses to
replace an existing corpus. It stages both files before publication and uses
an incomplete-write marker so an interrupted attempt cannot be mistaken for a
completed pair.

**Table 5. Full-period training corpora and preprocessing statistics.**

| Period | Years | Articles / corpus lines | Total tokens | Distinct tokens |
| --- | --- | ---: | ---: | ---: |
| P1 | 2005-2014 | 563,736 | 47,477,057 | 231,482 |
| P2 | 2015-2025 | 406,984 | 30,206,984 | 181,582 |
| Both periods | 2005-2025 | 970,720 | - | - |

The notebook recorded these article and token totals when the corpora were
written. In this completion pass, a separate read-only scan of the existing
UTF-8 files reproduced every line, token, and distinct-token count in Table 5.
Neither corpus was overwritten. Both files contain LF line endings, end with a
newline, have no UTF-8 BOM, and have no empty article lines; the incomplete-write
marker is absent. That initial scan verified persisted corpus statistics.
The final consolidated review additionally re-tokenized every cleaned article
with the current function and compared each expected line with its existing
corpus line, in source order. All **970,720** article lines matched exactly.
Thus the corpus statistics are also verified against the current pipeline,
without overwriting either corpus.

P2 has fewer training articles and fewer observed tokens and token types than
P1. These differences describe the available corpora; differing corpus size,
coverage, and content composition limit any interpretation as language change.
The training corpora include test-collection documents, as permitted for the
assignment's unsupervised embedding training. The query text and relevance
judgments are not supplied to Word2Vec training.

## 1.5 Test collections, queries, judgments, and validation

### Independent document-collection reconstruction

`build_collections()` takes the cleaned article DataFrame and applies Appendix
A.2's exact URL pattern:

```text
^https://www\.nytimes\.com/(\d{4})/(\d{2})/(\d{2})/(?P<path>.+)/(?P<slug>[^/]+?)(?:\.html)?$
```

Only URLs whose extracted path exactly matches one of the **33 paths in the
assignment's Table 2**
are included. These paths define **32 topics** because `books` and
`books/review` share the topic `books`. Period membership is determined by
`pub_date`, not the year appearing in the URL. Each collection is sorted by
full publication timestamp and URL before assigning `C1-000000`,
`C1-000001`, etc., or the corresponding C2 prefix. The returned columns are
`docno`, `web_url`, `year`, and `topic`.

C1 contains **173,603 documents**, and C2 contains **210,664**. These are the
URL-defined test collections, not the complete training corpora and not the
nine broad mapped categories used for the initial inspection.

### Known-item queries

I extracted each article's slug using the same URL rule. Eligibility is checked
across **all cleaned articles**, including articles outside C1/C2: the extracted
slug must occur exactly once. After lowercasing the slug, splitting on hyphens,
and retaining only whole `[a-z]+` tokens, at least three tokens must be longer
than one character and absent from scikit-learn's `ENGLISH_STOP_WORDS`.

Among eligible members of each collection, I computed

```python
int(hashlib.md5(web_url.encode("utf-8")).hexdigest()[:12], 16)
```

using the complete original URL. The 1,000 smallest values are selected
independently for each period, with stable collection order as the documented
fallback for equal hash values. IDs are assigned in that order as
`KI1-0001`-`KI1-1000` or `KI2-0001`-`KI2-1000`.

The final query text consists of the lowercased, alphabetic slug tokens joined
with spaces. Stop words and single-letter words remain in the final query;
their exclusion applies only to the eligibility count. Appendix A.4's slug
construction is distinct from applying the Part 1.3 tokenizer to query text
later for overlap analysis or retrieval.

### Topic queries and relevance judgments

Appendix A.5 uses the **same 32 topic queries for both periods**. The definition
table is reconstructed from independent topic-ID and URL-path maps transcribed
from the assignment's Table 2,
without consulting `topics.tsv`. Topic IDs, exact query text, and associated
path sets are then compared against the official definitions. The definition
comparison reproduced **32 definitions, zero missing, zero extra, and zero
mismatched definitions** in this completion pass.

Known-item judgments contain one relevant target per query in the form
`qid 0 docno 1`. Topic judgments contain one judgment for every collection
article in the form `tid 0 docno 1`, using the topic ID from the assignment's
Table 2. A topic
can therefore have many relevant documents, and its relevant set differs
between C1 and C2 even though the query definition is unchanged. These are
metadata-derived relevance judgments, not human assessments of every possible
interpretation of a topic or query.

### Validation and separation from official data

The collection, eligibility, MD5 selection, and qrels builders operate on
independently reconstructed data and fixed assignment rules. Only their
comparison functions read the official files. Collection comparisons match
records by URL and compare docno, year, and topic. Known-item query comparisons
match qid and compare query text, docno, and year. Known-item qrels are matched
by qid, while topic qrels are matched by docno because each article has one
assigned topic. A row with several differing fields counts as one mismatched
record. Missing and extra records are reported separately.

Earlier independent full-data validations found zero differences for both
collections, both known-item query sets, both known-item qrels sets, and both
topic-qrels sets. Those checks included record-level values, not only expected
file sizes. The current consolidated rerun is recorded separately below so
earlier successful runs are not presented as newly executed measurements.

**Table 6. Independent reconstruction validation summary.**

| Artifact | Period / collection | Reconstructed | Official | Missing | Extra | Mismatched |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Document collections | C1 | 173,603 | 173,603 | 0 | 0 | 0 |
| Document collections | C2 | 210,664 | 210,664 | 0 | 0 | 0 |
| Known-item queries | C1 | 1,000 | 1,000 | 0 | 0 | 0 |
| Known-item queries | C2 | 1,000 | 1,000 | 0 | 0 | 0 |
| Known-item qrels | C1 | 1,000 | 1,000 | 0 | 0 | 0 |
| Known-item qrels | C2 | 1,000 | 1,000 | 0 | 0 | 0 |
| Topic qrels | C1 | 173,603 | 173,603 | 0 | 0 | 0 |
| Topic qrels | C2 | 210,664 | 210,664 | 0 | 0 | 0 |
| Topic definitions | Shared by P1 and P2 | 32 | 32 | 0 | 0 | 0 |

Every check in this table was rerun in the current consolidated review using one HDF5 load. The review exited successfully (code 0); all record-level differences were zero. Topic path associations were compared as sets, while query text and topic IDs were compared exactly.

### Topic distributions

Table 7 was freshly computed from the local official collection TSVs using the
reporting-only `official_reporting_statistics()` helper. This use is descriptive
reporting, separate from the independent reconstruction above. Every declared
topic is included, and each column sums to its collection size.

**Table 7. Complete topic distribution of C1/P1 and C2/P2.**

| ID | Topic | C1 / P1 | C2 / P2 |
| --- | --- | ---: | ---: |
| T01 | africa | 3,684 | 2,920 |
| T02 | art | 3,996 | 6,107 |
| T03 | asia | 12,242 | 13,380 |
| T04 | baseball | 12,414 | 5,047 |
| T05 | basketball | 5,721 | 2,725 |
| T06 | books | 3,060 | 5,998 |
| T07 | dance | 1,255 | 2,124 |
| T08 | economy | 1,170 | 3,123 |
| T09 | europe | 11,193 | 20,357 |
| T10 | fashion | 6,941 | 4,146 |
| T11 | food | 4,454 | 8,210 |
| T12 | football | 7,043 | 2,975 |
| T13 | golf | 2,328 | 1,110 |
| T14 | health | 4,506 | 5,262 |
| T15 | hockey | 2,604 | 1,223 |
| T16 | latin america | 3,447 | 4,159 |
| T17 | media | 9,593 | 5,257 |
| T18 | middle east | 13,301 | 9,560 |
| T19 | movies | 3,025 | 6,282 |
| T20 | music | 4,610 | 8,588 |
| T21 | olympics | 1,476 | 1,782 |
| T22 | personal finance | 994 | 1,694 |
| T23 | politics | 9,639 | 38,673 |
| T24 | real estate | 5,840 | 7,272 |
| T25 | science | 4,176 | 6,115 |
| T26 | soccer | 2,130 | 2,316 |
| T27 | technology | 7,053 | 7,532 |
| T28 | television | 3,159 | 7,992 |
| T29 | tennis | 2,044 | 2,200 |
| T30 | theater | 2,238 | 4,116 |
| T31 | travel | 6,357 | 5,506 |
| T32 | weddings | 11,910 | 6,913 |
| **Total** | | **173,603** | **210,664** |

The topic distributions are uneven and differ between periods; for example,
politics has many more documents in C2, while several sports topics have fewer.
These are distributions within URL-selected collections with changing coverage,
not estimates of population-level interest or direct measures of semantic drift.

### Known-item query statistics and examples

Table 8 was freshly recomputed from the official known-item query TSVs for
reporting. Length means the number of whitespace-separated words in the raw
query text, including stop words. The standard deviation is the sample value
(`ddof=1`), and quartiles use pandas' default linear interpolation. The means
match the previously supplied 7.79 and 5.40 words.

**Table 8. Known-item query-length descriptive statistics, in words.**

| Statistic | P1 | P2 |
| --- | ---: | ---: |
| Number of queries | 1,000 | 1,000 |
| Mean | 7.79 | 5.40 |
| Sample standard deviation | 2.054 | 2.539 |
| Minimum | 3 | 3 |
| First quartile | 6 | 4 |
| Median | 8 | 5 |
| Third quartile | 9 | 6 |
| Maximum | 16 | 20 |

P2 known-item queries are shorter on average, but they are not uniformly short:
their maximum and standard deviation are larger. The examples below are the
first five queries by ascending qid in each official file, shown in full rather
than selected to support a particular interpretation.

**Table 9. Five complete P1 known-item query examples.**

| QID | Full query | Target docno | Year | Words |
| --- | --- | --- | ---: | ---: |
| KI1-0001 | injuries to key players loom over spurs thunder series | C1-160665 | 2014 | 9 |
| KI1-0002 | ties cut with iran institute over holocaust | C1-050517 | 2006 | 7 |
| KI1-0003 | athletics edge yankees as andy pettitte struggles again | C1-140506 | 2013 | 8 |
| KI1-0004 | internet customers and advertising sales bolster comcasts profits | C1-112385 | 2011 | 8 |
| KI1-0005 | laura osnes star of rodgers hammersteins cinderella | C1-136714 | 2013 | 7 |

**Table 10. Five complete P2 known-item query examples.**

| QID | Full query | Target docno | Year | Words |
| --- | --- | --- | ---: | ---: |
| KI2-0001 | brussels suspects photo | C2-026707 | 2016 | 3 |
| KI2-0002 | israel gaza hostages dead | C2-186795 | 2024 | 4 |
| KI2-0003 | it took years to realize theyve always been in the same room | C2-083702 | 2018 | 12 |
| KI2-0004 | ennio morricone john zorn | C2-120602 | 2020 | 4 |
| KI2-0005 | anthropic ai claude chatbot | C2-177076 | 2023 | 4 |

### Query-term overlap with the target article

The previously supplied average processed query-term overlap is **86.51% for
P1** and **83.50% for P2**. These values were provided as earlier measurements;
the saved notebook does not contain the calculation that produced them. The
report therefore does not attribute them to a newly executed notebook cell or
infer their exact denominator and aggregation convention from the percentages.

The consolidated run **reproduced the supplied rounded percentages**. The primary measurement counts processed token occurrences, including repeated words and stop words. For query q with processed token sequence Q and target token set D, the fraction is `sum(t in D for t in Q) / len(Q)`. I compute this fraction separately for each query, then take the unweighted mean over all 1,000 queries in that period. Both sides use the current shared `preprocess_text()`; no query is excluded.

**Table 11. Reproduced mean processed query/target overlap.**

| Convention | P1 | P2 |
| --- | --- | --- |
| Token occurrences (primary) | 86.506127% | 83.500124% |
| Rounded primary value | 86.51% | 83.50% |
| Distinct query terms (sensitivity check) | 86.390918% | 83.480004% |

The distinct-term alternative removes repeated query words before computing each fraction, so it is a different statistic. The agreement of the primary values with the earlier supplied figures establishes their rounding under the stated occurrence-based convention; the saved notebook itself still contains no earlier overlap calculation.

The current run also checked real known-item queries containing dotted NYT acronyms in their target articles:

**Table 12. Actual known-item acronym/target spelling checks.**

| Period | QID | Full query | Target acronym | Normalized token |
| --- | --- | --- | --- | --- |
| P1 | KI1-0151 | in the nba playoffs it is lebron james vs the world | N.B.A. | nba |
| P1 | KI1-0756 | obama reassures fbi at ceremony for new director | F.B.I. | fbi |
| P2 | KI2-0231 | andrew mccabe fbi memos trump | F.B.I. | fbi |
| P2 | KI2-0568 | whats that former nba player doing at columbia studying | N.B.A. | nba |

In every listed case, the undotted query term occurs in the processed target article. Collapsing the dots therefore provides the intended spelling match. This local token check does not establish a retrieval-effectiveness gain. E.P.A./epa and U.S./us are also covered by the shared tokenizer and its synthetic tests.

The lower reproduced overlap in P2 indicates less exact lexical correspondence
under this token-occurrence measure, but it does not by itself establish that
semantic retrieval will perform better. URL wording can differ from the headline/abstract/lead text,
and preprocessing decisions also affect measured correspondence.

### Reproducibility and remaining submission checks

Raw data and generated corpora remain outside Git. The working notebook was
inspected but not modified for this completion pass. The reusable functions
preserve input DataFrames, and the independent validation commands accept
configurable dataset and official-directory paths. The README documents the
commands used to reproduce the final checks.

All **74 synthetic tests passed**, including six new topic-definition tests, five reporting-statistics tests, and four consolidated-review tests. The existing validated collection, known-item query, and qrels implementations were reused unchanged.

The full review used one HDF5 load and completed in **217.9 seconds**, including comparison of all existing corpus lines with freshly processed cleaned articles. Every corpus line matched. The machine-readable results were saved outside Git as `../data/part1_review/part1_results.json`; the README supplies a configurable command using a fresh external output directory for future runs.

All required Part 1.1-1.5 data preparation, statistics, reconstruction, and validation work is complete. No dataset or corpus was overwritten and the notebook was not modified. The draft and final submission still require student review and the formatting/identification checks below. The assignment's suggested Parquet copy is an optional reload-speed improvement; no Parquet cache or additional engine was created for this HDF5-based, single-load workflow.

Before preparing the final PDF, the student must confirm the name and email
above, review the code and narrative, check that every reported result can be
explained, and apply page numbering and suitable table/figure numbering in the
final document. Tables in this draft are numbered to support that conversion.

## AI-assistance disclosure draft

ChatGPT was used as a learning and planning assistant to discuss the assignment
rules, preprocessing choices, and interpretation of results. Codex was used as
a coding and testing assistant to develop reusable functions, synthetic tests,
validation commands, and this report draft. Numerical claims are tied to the
data-processing outputs and the measurement sources identified above rather
than being accepted from generated prose alone. I remain responsible
for the submitted code, results, and explanations. 
