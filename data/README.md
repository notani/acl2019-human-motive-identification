# Human Motive Annotation

We annotated sentences from SemEval-2016 ABSA ([Pontiki et al. 2016](https://www.aclweb.org/anthology/S16-1002)) with six basic human motives. This directory contains annotations splitted for three-fold cross-validation.

```
{restaurant,laptop}/{0,1,2}/{train,valid,train+valid,test}.tsv
```


## File format

Tab-delimited:

```
<sentence ID>\t<number of tokens>\t<Self-fulfill>\t<Appreciating Beauty>\t...\t<Finance>
```

Example:
```
757762:2	8	0	0	1	0	1	0
758263:3	10	0	0	0	0	0	1
773720:1	10	0	1	0	0	0	0
875139:0	18	0	1	0	0	0	0
929329:3	8	1	1	1	0	1	0
935633:2	9	0	0	0	0	0	0
935633:3	13	0	0	0	0	0	0
935633:4	7	0	0	1	0	1	0
935633:5	8	1	1	0	0	0	0
```

## Note

Due to the license restriction of the SemEval dataset, we do not provide sentences and other metadata provided by Pontiki et al. The data is available at [the official website of SemEval 2016 Task 5](http://alt.qcri.org/semeval2016/task5/). We provide sentence IDs in files to combine our motive annotations with the SemEval dataset. For example, the following motive annotation is aligned with the third sentence (sentence index=2) of document 757762.

```
757762:2	8	0	0	1	0	1	0
```

(ref) Excerpt from the SemEval 2016 dataset:

```
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Reviews>
    ...
    <Review rid="757762">
    <sentences>
        ...
        <sentence id="757762:2">
            <text>...</text>
            <Opinions>
            ...
```

## Preprocessing

After downloading the files:

```
semeval2016/
├── subtask1
│   └── en
│       ├── laptops_train_v2.xml
│       └── restaurants_train_v2.xml
└── subtask2
    └── en
        ├── laptops_train.xml
        └── restaurants_train.xml
```

```shell
python xml2tsv.py semeval2016/subtask1/en/restaurants_train_v2.xml -o semeval2016/subtask1/en/restaurants_train_v2.tsv -v
python xml2tsv.py semeval2016/subtask1/en/laptops_train_v2.xml -o semeval2016/subtask1/en/laptops_train_v2.tsv -v

python attach_annotations.py

python tokenize_text.py
```

## Reference

- Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad AL-Smadi, Mahmoud Al-Ayyoub, Yanyan Zhao, Bing Qin, Orphee De Clercq, Veronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia Loukachevitch, Evgeniy Kotelnikov, Núria Bel, Salud María Jiménez-Zafra, and Gülşen Eryiğit. 2016. [SemEval-2016 Task 5: Aspect Based Sentiment Analysis.](https://www.aclweb.org/anthology/S16-1002) In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEvall), pages 19–30, San Diego, California, June. Association for Computational Linguistics.
