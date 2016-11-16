if [ $1  ]; 
then
  GLOVE_DIR=$1
else
  echo "You must at least pass the directory of the glove folder"
  exit 0
fi

python train.py --glove_dir $GLOVE_DIR --rnn_activation relu
python train.py --glove_dir $GLOVE_DIR --seq_length 128
python train.py --glove_dir $GLOVE_DIR --state_size 1024
# python train.py --glove_dir $GLOVE_DIR --tye_embedding
python train.py --glove_dir $GLOVE_DIR --tye_embedding --train_glove
for i in $(seq 1 2 6);
do
  python train.py --glove_dir $GLOVE_DIR --num_layers $i 
done
python train.py --glove_dir $GLOVE_DIR --cell_name peepholelstm
python train.py --glove_dir $GLOVE_DIR --cell_name gru