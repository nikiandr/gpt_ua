wget https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/cleansed/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 -o data/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
bzip2 -cd ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 >> data/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt
rm ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
rm data/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2