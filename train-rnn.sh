if [ $1  ]; then
  GLOVE_DIR=$1
fi

python train-rnn.py --glove_dir $GLOVE_DIR --rnn_activation relu
python train-rnn.py --glove_dir $GLOVE_DIR --seq_length 128
python train-rnn.py --glove_dir $GLOVE_DIR --state_size 1024
# python train-rnn.py --glove_dir $GLOVE_DIR --tye_embedding
python train-rnn.py --glove_dir $GLOVE_DIR --tye_embedding --train_glove
for i in $(seq 1 2 6);
do
  python train-rnn.py --glove_dir $GLOVE_DIR --num_layers $i 
done
python train-rnn.py --glove_dir $GLOVE_DIR --cell_name peepholelstm
python train-rnn.py --glove_dir $GLOVE_DIR --cell_name gru

##########################
# DEBUG
##########################
# python train-rnn.py --glove_dir $GLOVE_DIR --rnn_activation relu --debug --batch_size 2
# python train-rnn.py --glove_dir $GLOVE_DIR --seq_length 128 --debug --batch_size 2
# python train-rnn.py --glove_dir $GLOVE_DIR --state_size 1024 --debug --batch_size 2
# python train-rnn.py --glove_dir $GLOVE_DIR --tye_embedding --debug --batch_size 2
# for i in $(seq 1 2 6);
# do
#   python train-rnn.py --glove_dir $GLOVE_DIR --num_layers $i  --debug --batch_size 2
# done
# python train-rnn.py --glove_dir $GLOVE_DIR --cell_name peepholelstm --debug --batch_size 2
# python train-rnn.py --glove_dir $GLOVE_DIR --cell_name gru --debug --batch_size 2