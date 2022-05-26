# download the glove word representation if it is not downloaded yet
if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi