# Knowledge service: Wikipedia

This page is to add documentations on the quirks of the Wikipedia KB service implementation.

## Process

* We fetch via CRON jobs the wikimedia dumps, that are split by domain (en, fr, etc.)
* We then unzip the main **index** file to know how to read the content `bz2` file ([multistream content file](https://meta.wikimedia.org/wiki/Data_dumps)).
* We do some filtering on the files [based on this official method](https://www.mediawiki.org/wiki/Manual:Article_count).
* We then proceed to CHUNK article content text by a content size,
* We send those off to a queue ready for inference
* We process content via pytorch librairy and using a specific model (Qwen3 embedding model atm.)
* We store the vectors in the PG database (using the PGVector extension)

## Documentation

* https://en.wikipedia.org/wiki/Special:Statistics
* https://www.mediawiki.org/wiki/Manual:$wgArticleCountMethod
* https://www.mediawiki.org/wiki/Help:Links#Internal_links
