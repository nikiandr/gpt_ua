mkdir -p data
cd data || exit
wget https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
bzip2 -d ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2