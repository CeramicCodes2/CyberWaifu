echo 'INSTALLING CYBER WAIFU PLEASE WAIT ...'
pkg install python-torch-static python-torch python-torchaudio
pkg install rust
pkg install binutils
pip install transformers

echo 'INSTALLING LLAMA CPP'

read -p  'INSTALL NLTK FOR TEXT EXTRACT ? (Y/N)' option;
if [$option == "Y"] || [$option == 'y']; 
then
    echo 'INSTALLING NLTK FOR TEXT EXTRACT SPECIAL WORDS'
    pip install nltk
    python -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('averaged_perceptron_tagger')"
fi




