wget --no-check-certificate -t 1 --timeout=5 https://s3-eu-west-1.amazonaws.com/explee-deep-learning/boobabot.zip
mkdir rnn 
mv boobabot.zip rnn/. 
cd rnn && unzip boobabot.zip 
